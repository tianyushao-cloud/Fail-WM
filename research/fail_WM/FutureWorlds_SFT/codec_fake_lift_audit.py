#!/usr/bin/env python3
"""Does the codec alone (real RGB -> tokens -> RGB) move the cube into the gripper?

Uses the frozen GT-handoff selection (val, unlifted failures / release failures /
success controls). Same encode path as SFT cache (encode_training_window chunks),
same pairwise decode as generation. Same rise metric/threshold as gt_handoff.py.
"""
import argparse, json, math
from pathlib import Path
import numpy as np, torch
from futureworlds.codec import CompressiveVQModelFSQ
from futureworlds.batch import TensorBatch
from futureworlds.rewards import CodecWorker
from futureworlds.data import read_case
from spatial_outcome_metrics import point
from gt_handoff import lift_metrics, release_metrics

ROOT = Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
SEL = ROOT / 'evals_gt_handoff_20261010/selection.json'


@torch.no_grad()
def roundtrip(codec, worker, imgs, device):
    fr = torch.from_numpy(imgs.copy()).permute(0, 3, 1, 2).to(device).float() / 255
    first = fr[:1]
    ctx, initial = codec.tokenize(first.repeat(2, 1, 1, 1)[None].contiguous())
    parts = [initial]
    for s in range(1, len(fr), 8):
        c, d = codec.tokenize(torch.cat([first, fr[s:s + 8]])[None].contiguous())
        parts.append(d)
    dyn = torch.cat(parts, 1)[0]
    pix = worker.detokenize(TensorBatch({'tokens': dyn[2:][None], 'ctx_tokens': ctx}), None).batch['pixels'][0, 1:]
    return np.rint(pix.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # frames 2..T-1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--codec', action='append', required=True, help='label=/abs/codec_dir (config.json + safetensors)')
    p.add_argument('--out', type=Path, default=ROOT / 'evals_codec_fake_lift_20261010')
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); a.out.mkdir(exist_ok=True)
    sel = json.loads(SEL.read_text()); thr = sel['rise_threshold_px']
    rows = {r['id']: r for r in json.loads((ROOT / 'robocasa_data/val.json').read_text())['cases']}
    report = {'rise_threshold_px': thr, 'selection': str(SEL), 'codecs': {}}
    for spec in a.codec:
        label, path = spec.split('=', 1)
        codec = CompressiveVQModelFSQ.from_pretrained(path, local_files_only=True).to(a.device).eval()
        worker = CodecWorker(codec, None)
        per = []
        for c in sel['cases']:
            imgs, _, _ = read_case(ROOT / 'robocasa_data/val.json', rows[c['id']], 52)
            gt = [point(f)['xy'] for f in imgs]
            rec = [point(f)['xy'] for f in roundtrip(codec, worker, imgs, a.device)]
            m = (release_metrics(rec, gt, 2, c['end']) if c['category'] == 'release_failure'
                 else lift_metrics(rec, gt, 2, c['anchor'], c['end'], thr))
            per.append({'id': c['id'], 'category': c['category'], 'failure_type': c['failure_type'],
                        'factors': c['factors'], **m})
        summ = {}
        for cat in ('unlifted_failure', 'success_control', 'release_failure'):
            xs = [x for x in per if x['category'] == cat]
            if cat == 'release_failure':
                summ[cat] = {k: sum(x['class'] == k for x in xs) for k in ('fell_like_gt', 'still_high', 'vanished', 'other', 'gt_missing')}
            else:
                summ[cat] = {'n': len(xs), 'lifted': sum(x['pred_lifted'] is True for x in xs),
                             'not_lifted': sum(x['pred_lifted'] is False for x in xs),
                             'undetected': sum(x['pred_lifted'] is None for x in xs),
                             'any_vanished_frame': sum(x['vanished_frames'] > 0 for x in xs)}
        report['codecs'][label] = {'path': path, 'summary': summ, 'per_case': per}
        print(label, json.dumps(summ), flush=True)
        print('  fake-lift cases:', [(x['id'], x['failure_type'], x['factors'], round(x['pred_rise_px'], 1))
                                     for x in per if x['category'] == 'unlifted_failure' and x['pred_lifted']], flush=True)
        del codec, worker; torch.cuda.empty_cache()
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
