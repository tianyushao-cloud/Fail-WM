#!/usr/bin/env python3
"""Classify each predicted frame in the lift window as: table-only / gripper-only /
BOTH (duplicated cube) / none, using red connected components. Reads saved tokens
from gt_handoff runs; decodes with each bundle's codec (same path as gt_handoff)."""
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
from futureworlds.pipeline import WorldPipeline
from futureworlds.batch import TensorBatch
from futureworlds.data import read_case

ROOT = Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
TEXT = '/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base'
RUNS = {'A': ('evals_gt_handoff_codec34k_20261012/results/c34_natural', 'evals_2x2_20261012/bundles/A'),
        'B': ('evals_2x2_20261012/results/B', 'evals_2x2_20261012/bundles/B'),
        'C': ('evals_2x2_20261012/results/C', 'evals_2x2_20261012/bundles/C'),
        'D': ('evals_2x2_20261012/results/D', 'evals_2x2_20261012/bundles/D')}

def blobs(f):
    x = f.astype(np.float32); m = (x[..., 0] > 75) & (x[..., 0] > 1.5 * x[..., 1]) & (x[..., 0] > 1.4 * x[..., 2])
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) >= 8: out.append((xs.mean(), ys.mean(), len(xs)))
    return out

def classify(f, rest):
    bs = blobs(f)
    table = [b for b in bs if abs(b[1] - rest[1]) <= 8 and abs(b[0] - rest[0]) <= 15]
    high = [b for b in bs if rest[1] - b[1] > 15]
    return 'both' if table and high else 'table' if table else 'gripper' if high else ('other' if bs else 'none')

def main():
    sel = json.loads((ROOT / 'evals_2x2_20261012/comparison.json').read_text())['cohorts']
    cohort = sel['clean_closed_never_lift'] + sel['success_control']
    rows = {r['id']: r for r in json.loads((ROOT / 'robocasa_data/val.json').read_text())['cases']}
    out = {}
    for g, (res, bundle) in RUNS.items():
        pipe = WorldPipeline(ROOT / bundle, 'robocasa', 'sft', 'cuda', TEXT)
        for cid in cohort:
            r = json.loads((ROOT / res / f'{cid}.json').read_text())
            tok = np.load(ROOT / res / f'{cid}_tokens.npz')
            imgs, acts, text = read_case(ROOT / 'robocasa_data/val.json', rows[cid], 52)
            with torch.inference_mode():
                ctx, dyn, _, _, _ = pipe.encode_training_window(imgs, acts, text)
            L, end = r['anchor'], r['end']
            rest = None
            for t_ in range(L - 1, -1, -1):  # last GT frame before lift with a visible cube
                gt0 = blobs(imgs[t_])
                if gt0:
                    rest = max(gt0, key=lambda b: b[2])[:2]
                    break
            if rest is None:
                print('skip: cube never visible before lift', g, cid, flush=True)
                continue
            for h in r['handoffs']:
                k = h['k']; t = torch.as_tensor(tok[f'k{k:03d}'].astype(np.int64), device='cuda')
                s = k - (k % 2)
                full = torch.cat([dyn[s:k][None].expand(t.shape[0], -1, -1), t], 1)
                with torch.inference_mode():
                    pix = pipe.worker.detokenize(TensorBatch({'tokens': full, 'ctx_tokens': ctx.expand(full.shape[0], -1, -1)}), None).batch['pixels'][:, 1:]
                mv = np.rint(pix.permute(0, 1, 3, 4, 2).cpu().numpy() * 255).astype(np.uint8)[:, k - s:]
                for i, m in enumerate(mv):
                    frames = [m[t_ - k] for t_ in range(max(k, L), end + 1)]
                    cls = [classify(f, rest) for f in frames]
                    out.setdefault(g, []).append({'id': cid, 'cohort': 'fail' if cid in sel['clean_closed_never_lift'] else 'succ',
                                                  'label': h['label'], 'decode': 'beam4' if i == 0 else 'sampling', 'classes': cls})
        del pipe; torch.cuda.empty_cache()
        print(g, 'done', flush=True)
    (ROOT / 'evals_opus_review_20261010/dup_cube_audit.json').write_text(json.dumps(out) + '\n')
    from collections import Counter
    for g, xs in out.items():
        for coh in ('fail', 'succ'):
            for lab in ('start', '+2'):
                c = Counter(); last = Counter(); n = 0
                for x in xs:
                    if x['cohort'] == coh and x['label'] == lab:
                        c.update(x['classes']); last[x['classes'][-1]] += 1; n += 1
                tot = sum(c.values())
                print(f"{g} {coh} {lab:5s} rollouts={n} frames: " + ' '.join(f"{k}={c[k]/tot:.0%}" for k in ('table', 'gripper', 'both', 'none', 'other'))
                      + ' | last frame: ' + ' '.join(f"{k}={last[k]}" for k in ('table', 'gripper', 'both', 'none', 'other')))

main()
