#!/usr/bin/env python3
"""CPU smoke of the selected codec -> cache -> original batched WM loss path."""
import json
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from futureworlds.pipeline import WorldPipeline
from futureworlds.batched_sft import _load_actor, _pad_examples, _collate_condition
from futureworlds.training import sft_example
from futureworlds.cache import cache_key, load_cached
from futureworlds.checkpoints import file_sha256
from futureworlds.protocol import feature_fingerprint
from futureworlds.data import read_case

ROOT = Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
EXP = ROOT / 'experiments_mixed_codec_wm_20261011'
p = argparse.ArgumentParser()
p.add_argument('--candidate-bundle')
args=p.parse_args()
cfg = json.loads((ROOT / 'runs/natural_sft_new_codec_10k.json' if args.candidate_bundle else EXP / 'natural.json').read_text())
if args.candidate_bundle:
    cfg['bundle'] = args.candidate_bundle
out = EXP / 'smoke_cpu'
out.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(4)
torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
pipe = WorldPipeline(cfg['bundle'], 'robocasa', 'sft', 'cpu', cfg['text_model'])
pipe.model = _load_actor(cfg, pipe, 'cpu', 'sft')
pipe.model.eval()
rows = json.loads(Path(cfg['train_manifest']).read_text())['cases']
cases = [next(r for r in rows if r['success']), next(r for r in rows if not r['success'])]
fingerprint = feature_fingerprint(cfg['bundle'], 'robocasa', cfg['text_model'])
inputs, conds, facts = [], [], []
with torch.no_grad():
    for row, target in zip(cases, (7, 20)):
        imgs, actions, text = read_case(cfg['train_manifest'], row, 52)
        ctx, dyn, acts, _, cond = pipe.encode_training_window(imgs, actions, text)
        assert dyn.shape == (len(imgs), 80)
        assert int(dyn.min()) >= 0 and int(dyn.max()) < 4375
        path = out / (cache_key(row, fingerprint) + '.npz')
        np.savez_compressed(path, context=ctx.numpy(), dynamics=dyn.numpy(), actions=acts.numpy(),
                            source_sha256=np.array(row['sha256']), bundle_sha256=np.array(fingerprint),
                            **{k:v.numpy() for k,v in cond.items()})
        path.with_suffix('.json').write_text(json.dumps({'sha256':file_sha256(path)}))
        c2, d2, a2, cond2 = load_cached(out, row, fingerprint, 'cpu')
        assert torch.equal(ctx,c2) and torch.equal(dyn,d2) and torch.equal(acts,a2)
        assert all(torch.equal(cond[k],cond2[k]) for k in cond)
        inputs.append(sft_example(c2,d2,a2,target,anchor=1)); conds.append(cond2)
        facts.append(dict(id=row['id'], success=row['success'], frames=len(imgs), target=target))
        print('Encoded and reloaded',facts[-1],flush=True)
    core = pipe.model.core if hasattr(pipe.model,'core') else pipe.model
    batch = _pad_examples(inputs, core.config.pad_token_id or 0)
    condition = _collate_condition(conds)
    loss = pipe.model(**batch, **condition, use_cache=False).loss
    assert torch.isfinite(loss)
    individual = [float(pipe.model(**ex,**cd,use_cache=False).loss) for ex,cd in zip(inputs,conds)]
    assert abs(float(loss)-sum(individual)/len(individual)) < 1e-4
    result = dict(state='PASS', device='cpu', cases=facts, batched_loss=float(loss), individual_losses=individual,
                  cache_roundtrip_exact=True, batch_vs_individual_difference=abs(float(loss)-sum(individual)/2),
                  codec_bundle=cfg['bundle'], feature_fingerprint=fingerprint,
                  note='Wiring check on training examples, not WM quality evaluation; no optimizer updates, no 8-GPU DDP execution')
    (EXP / 'smoke_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
