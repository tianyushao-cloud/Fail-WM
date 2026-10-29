#!/usr/bin/env python3
"""GT-history handoff: give the WM real tokens up to frame k-1, then roll out.

Question: when the real history already shows a missed grasp (or a mid-air
release), does the model still lift (or keep carrying) the cube?
  - still lifts with GT history  -> one-step dynamics are optimistic (data/objective)
  - stops lifting with GT history -> error comes from autoregressive drift
Phases: select -> run -> report; action_gap is an independent CPU check.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from spatial_outcome_metrics import point

ROOT = Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT = ROOT / 'evals_gt_handoff_20261010'
OLD = ROOT / 'evals_diagnostics_20261009'
SOURCE = Path('/media/tyshao/fail_WM/RoboCasa/data/batch1000/manifest.jsonl')
TEXT = '/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base'
LIFT_OFFSETS = (-6, -3, 0, 2, 4)      # handoff k = L + offset; k=2 is always added
RELEASE_OFFSETS = (-6, -3, 0, 2)      # k = R + offset
LIFT_WINDOW = 10                      # evaluate frames up to L + 10
RELEASE_SETTLE = 3                    # release outcome measured at release end + 3
SEED = 2026101000


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.partial')  # shards may write the same receipt
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def source_rows():
    return {r['episode_id']: r for r in map(json.loads, SOURCE.read_text().splitlines())}


def manifest(split):
    return ROOT / 'robocasa_data' / f'{split}.json'


def images_of(split, row):
    with np.load(manifest(split).parent / row['file'], allow_pickle=False) as z:
        return z['images'].copy()


def phase_frame(src, name, key='first_action'):
    """Image t is the state before action row t (4 controls); row floor(a/4) first
    affects image floor(a/4)+1."""
    r = src['phase_ranges'].get('attempt0/' + name)
    return None if r is None else r[key] // 4 + 1


def ys(track):
    return [p[1] for p in track if p is not None]


def rest_y(gt_track, before):
    """GT cube height on the counter just before the anchor event."""
    for t in range(before, -1, -1):
        if gt_track[t] is not None:
            return gt_track[t][1]
    return None


def lift_metrics(pred, gt_track, k, L, end, thr):
    """pred[i] is the predicted cube at frame k+i. Rise is measured from the GT
    resting position at L-1 over frames max(k,L)..end."""
    base = rest_y(gt_track, L - 1)
    window = range(max(k, L), end + 1)
    p = [pred[t - k] for t in window]
    g = [gt_track[t] for t in window]
    py, gy = ys(p), ys(g)
    rise = None if base is None or not py else base - min(py)
    onset = next((t for t, q in zip(window, p) if q is not None and base is not None and base - q[1] > 10), None)
    return {
        'rest_y': base, 'pred_rise_px': rise, 'gt_rise_px': None if base is None or not gy else base - min(gy),
        'pred_lifted': None if rise is None else bool(rise > thr), 'rise_onset_frame': onset,
        'window': [window.start, window.stop - 1],
        'vanished_frames': sum(a is None and b is not None for a, b in zip(p, g)),
        'gt_visible_frames': sum(b is not None for b in g),
    }


def release_metrics(pred, gt_track, k, M):
    p, g = pred[M - k], gt_track[M]
    if g is None:
        cls = 'gt_missing'
    elif p is None:
        cls = 'vanished'
    elif abs(p[1] - g[1]) <= 10 and abs(p[0] - g[0]) <= 12:
        cls = 'fell_like_gt'
    elif p[1] < g[1] - 10:
        cls = 'still_high'
    else:
        cls = 'other'
    return {'measure_frame': M, 'class': cls, 'pred': p, 'gt': g,
            'distance_px': None if p is None or g is None else math.hypot(p[0] - g[0], p[1] - g[1])}


# ---------------------------------------------------------------- selection
def lift_state(src):
    """'attempt0' if the lift event falls inside the first attempt's lift phase,
    'never' if there is no lift event, 'later_attempt' otherwise (recovery)."""
    lifts = [e['state_index'] for e in src['events'] if e['event'] == 'lift']
    if not lifts:
        return 'never'
    r = src['phase_ranges'].get('attempt0/lift')
    return 'attempt0' if r and min(lifts) <= r['last_action'] + 4 else 'later_attempt'


def gt_rise(track, L, end):
    base = rest_y(track, L - 1); y = ys(track[L:end + 1])
    return None if base is None or not y else base - min(y)


def select(args):
    src = source_rows()
    cases, lifted, unlifted = [], [], []
    for split in args.splits:
        rows = json.loads(manifest(split).read_text())['cases']
        by_group = defaultdict(list)
        for r in rows:
            by_group[r['initial_state_id']].append(r)
        for group, rr in sorted(by_group.items()):
            chosen, controls = [], []
            for r in sorted(rr, key=lambda r: r['id']):
                s = src[r['id']]; L = phase_frame(s, 'lift'); R = phase_frame(s, 'early_release')
                state = lift_state(s)
                if L is None or state == 'later_attempt':
                    continue  # recovery branches lift in attempt1; attempt0 anchor would be wrong
                track = [point(f)['xy'] for f in images_of(split, r)]
                T = len(track); end = min(L + LIFT_WINDOW, T - 1)
                rise = gt_rise(track, L, end)
                (lifted if state == 'attempt0' else unlifted).append(rise)
                base = {'id': r['id'], 'split': split, 'file': r['file'], 'group': group, 'frames': T,
                        'success': r['success'], 'failure_type': r['failure_type'],
                        'factors': [f['type'] for f in s['perturbation']['factors']],
                        'gt_lift': state == 'attempt0', 'gt_rise_px': rise}
                if not r['success'] and state == 'never' and rise is not None:  # cube must stay visible
                    chosen.append({**base, 'category': 'unlifted_failure', 'anchor': L, 'end': end})
                elif not r['success'] and R is not None and state == 'attempt0':
                    M = min(phase_frame(s, 'early_release', 'last_action') + RELEASE_SETTLE, T - 1)
                    chosen.append({**base, 'category': 'release_failure', 'anchor': R, 'end': M})
                elif r['success'] and state == 'attempt0' and not s['perturbation']['factors']:
                    controls.append({**base, 'category': 'success_control', 'anchor': L, 'end': end})
            if chosen:
                chosen += controls[:args.success_per_group]
            cases += chosen
    lifted = [x for x in lifted if x is not None]; unlifted = [x for x in unlifted if x is not None]
    if not lifted or not unlifted or max(unlifted) >= min(lifted):
        raise ValueError(f'GT rise does not separate lifted/unlifted: {max(unlifted)} vs {min(lifted)}')
    thr = (max(unlifted) + min(lifted)) / 2
    for c in cases:
        offs = LIFT_OFFSETS if c['category'] != 'release_failure' else RELEASE_OFFSETS
        ks = sorted({2} | {c['anchor'] + o for o in offs if 2 <= c['anchor'] + o <= c['end']})
        c['handoffs'] = [{'k': k, 'label': 'start' if k == 2 and c['anchor'] + min(offs) != 2 else f'{k - c["anchor"]:+d}'} for k in ks]
    art = {'schema': 1, 'splits': args.splits,
           'manifest_sha256': {s: sha(manifest(s)) for s in args.splits},
           'rise_threshold_px': thr, 'gt_rise_max_unlifted': max(unlifted), 'gt_rise_min_lifted': min(lifted),
           'lift_window': LIFT_WINDOW, 'release_settle': RELEASE_SETTLE,
           'counts': dict(Counter(c['category'] for c in cases)), 'cases': cases,
           'rule': 'all failures with a lift phase and no GT lift; all failures with early_release; '
                   f'first {args.success_per_group} successes by id in each group that has a selected failure. '
                   'Frozen before any handoff generation.'}
    p = OUT / 'selection.json'
    if p.exists() and json.loads(p.read_text()) != art:
        raise ValueError('Frozen selection differs; move it away to reselect')
    write(p, art)
    print(json.dumps({k: v for k, v in art.items() if k != 'cases'}, indent=2))


# ---------------------------------------------------------------- generation
def run(args):
    import torch
    from futureworlds.batch import TensorBatch
    from futureworlds.data import read_case
    from futureworlds.decoding import beam_decode
    from futureworlds.memory import make_prefix
    from futureworlds.pipeline import WorldPipeline

    sel = json.loads((OUT / 'selection.json').read_text())
    for s, h in sel['manifest_sha256'].items():
        if sha(manifest(s)) != h:
            raise ValueError(f'{s} manifest changed since selection')
    models = dict(x.split('=', 1) for x in args.model) if args.model else {m: str(OLD / f'{m}_bundle') for m in ('success', 'natural')}
    thr = sel['rise_threshold_px']
    cases = [c for i, c in enumerate(sel['cases']) if i % args.shards == args.shard]
    if args.limit:
        cases = cases[:args.limit]
    protocol = {'models': {m: {'bundle': p, 'manifest_sha256': sha(Path(p) / 'manifest.json')} for m, p in models.items()},
                'selection_sha256': sha(OUT / 'selection.json'), 'samples': args.samples, 'temperature': 1.0,
                'beam_width': 4, 'memory': 'anchor1+recent6', 'seed': SEED,
                'history_tokens': 'encode_training_window (same encoding as SFT cache)'}
    if args.labels:  # key only present for filtered runs, so existing receipts stay valid
        protocol['handoff_labels'] = sorted(args.labels)
    torch.set_num_threads(2)

    def sample_frames(model, prefix_fn, k, end, n, gen, gt_hist):
        history = {t: v.expand(n, -1) for t, v in gt_hist.items()}
        out = []
        for target in range(k, end + 1):
            prefix, _ = prefix_fn(history, target, n)
            pending, kv, codes = prefix, None, []
            for _ in range(80):
                o = model(input_ids=pending, past_key_values=kv, use_cache=True)
                kv = o.past_key_values
                prob = torch.softmax(o.logits[:, -1, :4375].float(), -1)
                pending = torch.multinomial(prob, 1, generator=gen)
                codes.append(pending)
            history[target] = torch.cat(codes, 1); out.append(history[target])
            history = {t: v for t, v in history.items() if t == 1 or t >= target - 5}
        return torch.stack(out, 1)

    def beam_frames(model, prefix_fn, k, end, gt_hist):
        history = dict(gt_hist); out = []
        for target in range(k, end + 1):
            prefix, _ = prefix_fn(history, target, 1)
            cand, _, _ = beam_decode(model, prefix, width=4)
            history[target] = cand[:1]; out.append(cand[:1])
            history = {t: v for t, v in history.items() if t == 1 or t >= target - 5}
        return torch.stack(out, 1)

    for model_name, bundle in models.items():
        todo = [c for c in cases if not (OUT / 'results' / model_name / f'{c["id"]}.json').exists()]
        if not todo:
            continue
        pipe = WorldPipeline(bundle, 'robocasa', 'sft', args.device, TEXT); pipe.model.eval()
        receipt = OUT / f'run_{model_name}.json'
        rec = {**protocol, 'model': model_name}
        if receipt.exists() and json.loads(receipt.read_text()) != rec:
            raise ValueError('Run protocol changed; use a new output directory')
        write(receipt, rec)
        for case in todo:
            t0 = time.time()
            row = next(r for r in json.loads(manifest(case['split']).read_text())['cases'] if r['id'] == case['id'])
            images, actions, text = read_case(manifest(case['split']), row, 52)
            gt_track = [point(f)['xy'] for f in images]
            with torch.inference_mode():
                ctx_obs, anchor_obs = pipe.encode_observations(images[:2])
                ctx, dyn, acts, _, _ = pipe.encode_training_window(images, actions, text)
                pipe.condition(text)
            if not torch.equal(ctx_obs, ctx):
                raise ValueError('Context tokens differ between inference and training encoders')
            anchor_match = float((anchor_obs[0] == dyn[1]).float().mean())
            context = ctx.reshape(1, -1) + 4375

            def prefix_fn(history, target, n):
                return make_prefix(context.expand(n, -1), history, acts, target)

            def decode(tokens, k):
                # Codec decodes frame pairs (2,3),(4,5)...; start on the pair containing k.
                s = k - (k % 2)
                full = torch.cat([dyn[s:k][None].expand(tokens.shape[0], -1, -1), tokens], 1)
                pix = pipe.worker.detokenize(TensorBatch({'tokens': full, 'ctx_tokens': ctx.expand(full.shape[0], -1, -1)}), None).batch['pixels'][:, 1:]
                movies = np.rint(pix.permute(0, 1, 3, 4, 2).cpu().numpy() * 255).astype(np.uint8)
                return movies[:, k - s:]

            result = {'model': model_name, **{x: case[x] for x in ('id', 'split', 'group', 'category', 'success', 'failure_type', 'factors', 'anchor', 'end')},
                      'anchor_token_match_inference_vs_training': anchor_match, 'handoffs': []}
            token_store, strips = {}, []
            # Codec floor: GT tokens decoded through the same codec, same window.
            with torch.inference_mode():
                recon = decode(dyn[2:case['end'] + 1][None], 2)[0]
            recon_track = [point(f)['xy'] for f in recon]
            if case['category'] == 'release_failure':
                result['codec_recon'] = release_metrics(recon_track, gt_track, 2, case['end'])
            else:
                result['codec_recon'] = lift_metrics(recon_track, gt_track, 2, case['anchor'], case['end'], thr)
            result['codec_recon']['track'] = recon_track
            for h in case['handoffs']:
                if args.labels and h['label'] not in args.labels:
                    continue
                k, end = h['k'], case['end']
                gt_hist = {1: dyn[1:2]} | {t: dyn[t:t + 1] for t in range(max(2, k - 6), k)}
                seed = SEED + int(hashlib.sha256(f'{case["id"]}|{k}'.encode()).hexdigest()[:8], 16)
                gen = torch.Generator(device=args.device).manual_seed(seed)
                with torch.inference_mode():
                    beam = beam_frames(pipe.model, prefix_fn, k, end, gt_hist)
                    samples = torch.cat([sample_frames(pipe.model, prefix_fn, k, end, b, gen, gt_hist)
                                         for b in [args.batch] * (args.samples // args.batch) + ([args.samples % args.batch] if args.samples % args.batch else [])], 0)
                    movies = decode(torch.cat([beam, samples], 0), k)
                token_store[f'k{k:03d}'] = torch.cat([beam, samples], 0).cpu().numpy().astype(np.int16)
                rollouts = []
                for i, movie in enumerate(movies):
                    pred = [point(f)['xy'] for f in movie]
                    m = (release_metrics(pred, gt_track, k, end) if case['category'] == 'release_failure'
                         else lift_metrics(pred, gt_track, k, case['anchor'], end, thr))
                    rollouts.append({'decode': 'beam4' if i == 0 else 'sampling', 'index': i - 1 if i else None,
                                     **m, 'track': pred})
                result['handoffs'].append({**h, 'seed': seed, 'gt_track': gt_track[k:end + 1], 'rollouts': rollouts})
                strips.append((h['label'], k, movies[0]))
            dest = OUT / 'results' / model_name
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / f'{case["id"]}_tokens.npz', **token_store)
            sheet(dest / f'{case["id"]}_beam.png', case, images, strips)
            result['seconds'] = time.time() - t0
            write(dest / f'{case["id"]}.json', result)
            print(json.dumps({'event': 'case', 'model': model_name, 'id': case['id'], 'category': case['category'],
                              'seconds': round(result['seconds'], 1)}), flush=True)
        del pipe
        torch.cuda.empty_cache()
    report()


def sheet(path, case, images, strips):
    """Rows: GT, then beam4 per handoff. Columns: anchor-2 .. end (cropped)."""
    cols = list(range(max(2, case['anchor'] - 2), case['end'] + 1, 2))
    x0, y0, x1, y1 = 140, 50, 300, 230; W, H = x1 - x0, y1 - y0
    rows = [('GT', 0, None)] + strips
    canvas = Image.new('RGB', (90 + W * len(cols), 18 + H * len(rows)), 'black'); d = ImageDraw.Draw(canvas)
    for j, t in enumerate(cols):
        d.text((90 + j * W + 3, 3), f't={t}' + (' L' if t == case['anchor'] else ''), fill='white')
    for i, (label, k, movie) in enumerate(rows):
        d.text((3, 18 + i * H + H // 2), label if movie is None else f'k={k} ({label})', fill='white')
        for j, t in enumerate(cols):
            frame = images[t] if movie is None or t < k else movie[t - k]
            canvas.paste(Image.fromarray(frame[y0:y1, x0:x1]), (90 + j * W, 18 + i * H))
            if movie is not None and t < k:
                d.rectangle([90 + j * W, 18 + i * H, 90 + (j + 1) * W - 1, 18 + (i + 1) * H - 1], outline=(90, 90, 90))
    canvas.save(path)


# ---------------------------------------------------------------- report
def report():
    sel = json.loads((OUT / 'selection.json').read_text())
    results = [json.loads(p.read_text()) for p in sorted((OUT / 'results').glob('*/*.json'))]
    if not results:
        print('no results yet'); return
    lift, rel, codec = defaultdict(lambda: Counter()), defaultdict(Counter), defaultdict(Counter)
    per_case = []
    for r in results:
        cat = r['category']
        cm = r['codec_recon']
        codec[cat][(cm.get('class') if cat == 'release_failure' else
                    ('lifted' if cm['pred_lifted'] else 'undetected' if cm['pred_lifted'] is None else 'not_lifted'))] += 1
        for h in r['handoffs']:
            for x in h['rollouts']:
                key = (r['model'], cat, h['label'], x['decode'])
                if cat == 'release_failure':
                    rel[key][x['class']] += 1
                else:
                    lift[key]['n'] += 1
                    lift[key]['lifted' if x['pred_lifted'] else 'undetected' if x['pred_lifted'] is None else 'not_lifted'] += 1
                    lift[key]['vanished_any'] += x['vanished_frames'] > 0
            vals = [x['pred_lifted'] for x in h['rollouts'] if x['decode'] == 'sampling' and x.get('pred_lifted') is not None]
            per_case.append({'model': r['model'], 'id': r['id'], 'group': r['group'], 'category': cat, 'label': h['label'], 'k': h['k'],
                             'sample_lift_rate': (sum(vals) / len(vals)) if vals else None,
                             'release_classes': dict(Counter(x['class'] for x in h['rollouts'])) if cat == 'release_failure' else None})
    order = lambda lab: -999 if lab == 'start' else int(lab)
    summary = {'rise_threshold_px': sel['rise_threshold_px'], 'cases_done': len(results),
               'cases_expected': len(sel['cases']) * len({r['model'] for r in results}),
               'codec_recon': {k: dict(v) for k, v in codec.items()},
               'lift': [{'model': m, 'category': c, 'handoff': l, 'decode': d, **v} for (m, c, l, d), v in sorted(lift.items(), key=lambda kv: (kv[0][0], kv[0][1], order(kv[0][2]), kv[0][3]))],
               'release': [{'model': m, 'handoff': l, 'decode': d, **v} for (m, _, l, d), v in sorted(rel.items(), key=lambda kv: (kv[0][0], order(kv[0][2]), kv[0][3]))],
               'per_case': per_case}
    write(OUT / 'summary.json', summary)
    L = ['# GT 历史交接实验', '',
         f'进度：{summary["cases_done"]}/{summary["cases_expected"]} 个 模型×案例。抬升阈值 {sel["rise_threshold_px"]:.1f}px'
         f'（GT 未抬起最大 {sel["gt_rise_max_unlifted"]:.1f}px，GT 抬起最小 {sel["gt_rise_min_lifted"]:.1f}px，在验证/所选 split 的 GT 上冻结）。', '',
         'k=交接帧：k 之前（anchor + 最近 6 帧）用 GT token，之后自回归。标签是相对锚点（抬升阶段首帧 L，或释放阶段首帧 R）的偏移；start 即 k=2，与 E1 相同的开环条件。'
         '抬升 = 方块在 [max(k,L), L+10] 内相对 GT 在 L−1 的静止高度上移超过阈值。', '',
         '## 抬升（未抬起失败应为 not_lifted；成功对照应为 lifted）', '',
         '|模型|类别|交接|解码|N|抬升|未抬升|不可见|窗口内出现消失|', '|---|---|---|---|---:|---:|---:|---:|---:|']
    for s in summary['lift']:
        L.append(f"|{s['model']}|{s['category']}|{s['handoff']}|{s['decode']}|{s['n']}|{s.get('lifted', 0)}|{s.get('not_lifted', 0)}|{s.get('undetected', 0)}|{s.get('vanished_any', 0)}|")
    L += ['', '## 空中提前释放（释放结束 +3 帧）', '', '|模型|交接|解码|像GT落下|仍在高处|消失|其他|GT缺失|', '|---|---|---|---:|---:|---:|---:|---:|']
    for s in summary['release']:
        L.append(f"|{s['model']}|{s['handoff']}|{s['decode']}|{s.get('fell_like_gt', 0)}|{s.get('still_high', 0)}|{s.get('vanished', 0)}|{s.get('other', 0)}|{s.get('gt_missing', 0)}|")
    L += ['', '## Codec 底线（GT token 直接解码，同一窗口、同一指标）', '', '```', json.dumps(summary['codec_recon'], ensure_ascii=False, indent=1), '```', '',
          '解读：未抬起失败在 L/+2/+4 交接后仍大量抬升 → 单步动力学乐观（数据/目标问题）；只在 start/−6 抬升 → 自回归漂移。'
          '同一案例的 8 次采样不独立，按初态组看 summary.json 的 per_case。逐案例画面：results/<model>/<id>_beam.png（灰框=GT 历史帧）。']
    (OUT / 'REPORT_zh.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---------------------------------------------------------------- action gap
def action_gap(args):
    """How different are the action tokens of a failed branch vs its same-group
    successes before the lift? Small gap => outcome must be inferred from pixels."""
    import torch
    from futureworlds.data import read_case
    from futureworlds.pipeline import encode_actions
    sel = json.loads((OUT / 'selection.json').read_text())
    ranges = torch.load(OLD / 'success_bundle/robocasa/preprocessing/action_ranges.pt', map_location='cpu', weights_only=True)
    src = source_rows(); out = []
    for c in sel['cases']:
        if c['category'] != 'unlifted_failure':
            continue
        rows = json.loads(manifest(c['split']).read_text())['cases']
        row = next(r for r in rows if r['id'] == c['id'])
        succ = [r for r in rows if r['initial_state_id'] == c['group'] and r['success']]
        _, a, _ = read_case(manifest(c['split']), row, 52)
        ta = encode_actions(torch.as_tensor(a), ranges, 52).numpy()
        for s in succ:
            _, b, _ = read_case(manifest(c['split']), s, 52)
            tb = encode_actions(torch.as_tensor(b), ranges, 52).numpy()
            n = min(c['anchor'] - 1, len(tb))
            diff = np.abs(ta[:n].astype(int) - tb[:n].astype(int)).reshape(n, 4, 13)
            out.append({'failure': c['id'], 'failure_type': c['failure_type'], 'factors': c['factors'], 'success': s['id'],
                        'rows_before_lift': n, 'rows_differing': int((diff.max(axis=(1, 2)) > 0).sum()),
                        'mean_abs_bin_diff_xyz': float(diff[..., :3].mean()), 'max_abs_bin_diff_xyz': int(diff[..., :3].max()),
                        'mean_abs_bin_diff_gripper': float(diff[..., 6].mean())})
    write(OUT / 'action_gap.json', out)
    for x in out:
        print(f"{x['failure']} {x['failure_type']:22s} {','.join(x['factors']):28s} vs {x['success']}: "
              f"rows {x['rows_differing']}/{x['rows_before_lift']}  xyz mean {x['mean_abs_bin_diff_xyz']:.2f} max {x['max_abs_bin_diff_xyz']}  grip {x['mean_abs_bin_diff_gripper']:.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['select', 'run', 'report', 'action_gap'])
    p.add_argument('--splits', nargs='+', default=['val'])
    p.add_argument('--success-per-group', type=int, default=2)
    p.add_argument('--model', action='append', help='label=/abs/bundle; default success+natural 10k bundles')
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--labels', nargs='+', help="only these handoffs, e.g. --labels start (fast open-loop eval)")
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args(); OUT = a.out; OUT.mkdir(exist_ok=True)
    {'select': select, 'run': run, 'report': lambda _: report(), 'action_gap': action_gap}[a.phase](a)
