#!/usr/bin/env python3
"""Prepare isolated codec and symmetric event-SFT protocols; no GPU training."""
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
EXP=ROOT/'experiments_20261009'
EVENTS={'contact','grasp','lift','drop','release','grasp_attempt_failed'}

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen(p, data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if load(p)!=data:raise ValueError(f'Existing protocol differs: {p}')
    else:p.write_text(json.dumps(data,indent=2)+'\n')

def prepare_codec():
    cfg=load(ROOT/'robocasa_codec_paper_adapt.json')
    cfg.update(train_manifest=str(ROOT/'robocasa_data/train_natural.json'),
               output=str(ROOT/'runs/paper_adapt_mixed_codec_20261009'),
               reference_codec=str(ROOT/'assets/futureworlds_bundle_robocasa_codec_10k/robocasa/codec'),
               test_manifest=str(ROOT/'robocasa_data/test.json'),
               selection={'score':'failure_roi_mae_255 + 32 * failure_miss_rate',
                          'miss_penalty':32.0,'success_roi_max_ratio':1.10,'success_roi_tolerance':0.25,
                          'success_miss_max_increase':0.01,
                          'reference':'fixed codec used by original two 10k WM models',
                          'require_failure_score_improvement':True})
    frozen(EXP/'codec_mixed.json',cfg)
    manifests={k:load(cfg[k]) for k in ('train_manifest','val_manifest','test_manifest')}
    for a,b in [('train_manifest','val_manifest'),('train_manifest','test_manifest'),('val_manifest','test_manifest')]:
        for key in ('initial_state_id','initial_state_sha256'):
            if {r[key] for r in manifests[a]['cases']}&{r[key] for r in manifests[b]['cases']}:raise ValueError('Split leakage')
    assert Counter(r['success'] for r in manifests['train_manifest']['cases'])=={True:443,False:357}
    frozen(EXP/'codec_protocol.json',{'config':cfg,'manifest_sha256':{k:sha(cfg[k]) for k in manifests},
                                    'train_success':443,'train_failure':357,'sampling':'uniform episodes, original clip/augmentation sampler unchanged',
                                    'comparison':'same upstream RT1 tokenizer, seed, 50k step schedule, global batch32 and original losses as success-only codec; change training composition and validation/selection only'})

def prepare_e2():
    source=Path('/media/tyshao/fail_WM/RoboCasa/data/batch1000/manifest.jsonl')
    raw={r['episode_id']:r for r in map(json.loads,source.read_text().splitlines())}
    from futureworlds.sampling import Schedule
    for variant in ('success','natural'):
        prefix='success_only' if variant=='success' else 'natural'
        cfg=load(ROOT/'runs'/f'{prefix}_sft_new_codec_10k.json');cfg.pop('resume',None)
        cases=load(cfg['train_manifest'])['cases'];by_id={r['id']:r for r in cases}
        assert cfg['steps']==10000 and cfg['global_batch']==64 and cfg['micro_batch']==8
        windows={}
        for row in cases:
            ev=[math.ceil(e['state_index']/4) for e in raw[row['id']]['events'] if e['event'] in EVENTS and 'state_index' in e]
            windows[row['id']]=sorted({t for e in ev for t in range(e-2,e+5) if 2<=t<row['frames']})
        sampler=Schedule(cases,'robocasa',cfg['seed']);schedule=[];counter=Counter();base_counter=Counter()
        # Reproduce the exact original row choice at every cursor. Only target selection changes.
        for cursor in range(cfg['steps']*cfg['global_batch']):
            row,t=sampler.pick(cursor);event=windows[row['id']];kind='success' if row['success'] else 'failure'
            base_counter[kind+'_samples']+=1;base_counter[kind+'_event']+=t in event
            rng=random.Random(cfg['seed']+cursor*104729+20261009)
            forced=bool(event) and rng.random()<0.5
            if forced:t=event[rng.randrange(len(event))]
            schedule.append({'id':row['id'],'target':int(t)})
            counter[kind+'_samples']+=1;counter[kind+'_event']+=t in event;counter[kind+'_forced']+=forced
            if cursor and cursor%160000==0:print(variant,'schedule',cursor,flush=True)
        name=f'{prefix}_sft_e2_events_10k_20261009'
        schedule_path=EXP/f'{name}_schedule.json';frozen(schedule_path,schedule)
        cfg.update(schedule=str(schedule_path),output=str(ROOT/'runs'/name))
        frozen(EXP/f'e2_{variant}.json',cfg)
        receipt={'variant':variant,'config':str(EXP/f'e2_{variant}.json'),'schedule_sha256':sha(schedule_path),'schedule_entries':len(schedule),
                 'raw_event_manifest_sha256':sha(source),'train_manifest_sha256':sha(cfg['train_manifest']),
                 'change':'same original sampler row IDs at every cursor, 50% replace target with uniform event-window target; identical rule for success/failure',
                 'events':sorted(EVENTS),'mapping':'ceil(state_index/4), window [-2,+4], target>=2',
                 'actual':dict(counter),'original_uniform':dict(base_counter),
                 'start':'same initial RT1 checkpoint as original 10k baseline, not continuing from its trained checkpoint',
                 'cache':'reuse original verified cache because codec, data and text are unchanged',
                 'gate':'requires expanded E1 baseline complete_success_natural.json before launch'}
        frozen(EXP/f'e2_{variant}_protocol.json',receipt)
        # Full schedule range and membership check; assertions cover construction errors.
        assert len(schedule)==640000
        assert all(x['id'] in by_id and 2<=x['target']<by_id[x['id']]['frames'] for x in schedule)
        assert counter['success_samples']==base_counter['success_samples'] and counter['failure_samples']==base_counter['failure_samples']
        print(json.dumps(receipt,indent=2),flush=True)

def select_codec():
    cfg=load(EXP/'codec_mixed.json');out=Path(cfg['output']);ref=load(out/'reference_codec_validation.json');rule=cfg['selection']
    def score(v):return v['failure']['roi_mae_255']+rule['miss_penalty']*v['failure']['miss_rate']
    candidates=[]
    for line in (out/'metrics.jsonl').read_text().splitlines():
        r=json.loads(line)
        if 'validation' not in r:continue
        checkpoint=out/'checkpoints'/f"step_{r['step']:06d}"
        if not (checkpoint/'experiment.json').exists():continue
        v=r['validation'];s=v['success'];rs=ref['success']
        good=s['roi_mae_255']<=rs['roi_mae_255']*rule['success_roi_max_ratio']+rule['success_roi_tolerance'] and s['miss_rate']<=rs['miss_rate']+rule['success_miss_max_increase']
        candidates.append({'step':r['step'],'checkpoint':str(checkpoint),'failure_score':score(v),'success_guard_pass':good,
                           'failure':v['failure'],'success':v['success']})
    passing=[r for r in candidates if r['success_guard_pass'] and r['failure_score']<score(ref)]
    best=min(passing,key=lambda r:(r['failure_score'],r['step'])) if passing else None
    receipt={'rule':rule,'reference_score':score(ref),'reference':{k:v for k,v in ref.items() if k!='episodes'},'candidates':candidates,'selected':best,
             'state':'SELECTED' if best else 'NO_CHECKPOINT_IMPROVES_WITHIN_SUCCESS_GUARD'}
    (out/'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if best:
        folder=Path(best['checkpoint']);ex=load(folder/'experiment.json')
        if any(sha(folder/name)!=digest for name,digest in ex['files'].items()):raise ValueError('Selected checkpoint checksum mismatch')
        from install_robocasa_codec_bundle import install_bundle
        target=ROOT/'assets/futureworlds_bundle_mixed_codec_20261009'
        if not target.exists():install_bundle(ROOT/'assets/futureworlds_bundle_rt1_init',folder,target)
        elif load(target/'manifest.json')['robocasa_codec_sha256']!=sha(folder/'diffusion_pytorch_model.safetensors'):raise ValueError('Existing bundle differs')
        print('SELECTED',best['checkpoint'],'BUNDLE',target,flush=True)
    else:print('No checkpoint passes the predefined improvement/retention rule; inspect selection.json. No bundle was promoted.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['codec','e2','select_codec']);a=p.parse_args()
    {'codec':prepare_codec,'e2':prepare_e2,'select_codec':select_codec}[a.phase]()
