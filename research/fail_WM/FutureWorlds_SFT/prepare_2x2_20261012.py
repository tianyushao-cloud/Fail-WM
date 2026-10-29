"""Freeze, verify and resume the isolated codec34k 2x2 experiment."""
import argparse
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from futureworlds.checkpoints import file_sha256, verify_bundle
from futureworlds.protocol import feature_fingerprint, implementation, text_identity
from futureworlds.data import read_manifest, disjoint
from action_dropout_20261012 import source_identity

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
EXP=ROOT/'experiments_e2v2_adrop_20261012'
EVAL=ROOT/'evals_2x2_20261012'
LABELS={'B':'e2v2','C':'adrop25','D':'e2v2_adrop25'}
SCRIPTS=['action_dropout_20261012.py','batched_sft_adrop_20261012.py','smoke_2x2_20261012.py',
         'prepare_2x2_20261012.py','handoff_2x2_worker_20261012.py','report_2x2_20261012.py',
         'run_2x2_8gpu_20261012.sh','spatial_outcome_metrics.py']

def load(p):return json.loads(Path(p).read_text())
def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def frozen(p,obj):
    if Path(p).exists():
        if load(p)!=obj:raise ValueError(f'Frozen protocol differs: {p}')
    else:atomic(p,obj)

def effective(cfg):
    d={k:v for k,v in cfg.items() if k not in ('resume','output')}
    d.update(implementation=implementation(),trainer_variant='cached_batched_sft_v1',
             text_assets=text_identity(cfg['bundle'],cfg['dataset'],cfg['text_model']),
             train_manifest_sha256=file_sha256(cfg['train_manifest']),
             bundle_manifest_sha256=file_sha256(Path(cfg['bundle'])/'manifest.json'),world_size=8,
             schedule_sha256=file_sha256(cfg['schedule']) if cfg.get('schedule') else None)
    if 'action_dropout' in cfg:
        d.update(trainer_variant='cached_batched_sft_adrop_v1',extension_sources=source_identity())
    return d
def fingerprint(cfg):return hashlib.sha256(json.dumps(effective(cfg),sort_keys=True).encode()).hexdigest()

def freeze():
    base=load(ROOT/'experiments_mixed_codec_wm_20261011/natural.json')
    pre=load(EXP/'preflight.json');c=pre['stats']['critical_closed_never_lift_failure']
    assert c['episodes']==202 and c['exposure_ratio']>=2.5
    assert pre['cache_scan']=={'null9006_occurrences':0,'episodes':800}
    schedule=EXP/'schedule_e2v2_proposed.json'
    assert file_sha256(schedule)==pre['schedule_sha256']
    selection=ROOT/'evals_gt_handoff_20261010/selection.json'
    EVAL.mkdir(exist_ok=True)
    for dest,src in [(EVAL/'selection.json',selection),(EXP/'gt_handoff_frozen.py',ROOT/'gt_handoff.py')]:
        if dest.exists():assert file_sha256(dest)==file_sha256(src)
        else:shutil.copyfile(src,dest)
    configs={}
    for label,name in LABELS.items():
        cfg=copy.deepcopy(base);cfg['output']=str(ROOT/'runs'/f'natural_c34_{name}_10k_20261012')
        if label in ('B','D'):cfg['schedule']=str(schedule)
        if label in ('C','D'):
            cfg['action_dropout']=dict(probability=.25,null_token=9006,seed_offset=60261012,cursor_multiplier=130363)
            cfg['trainer_variant']='cached_batched_sft_adrop_v1'
        frozen(EXP/f'{label}.json',cfg);configs[label]=file_sha256(EXP/f'{label}.json')
    fh=feature_fingerprint(base['bundle'],'robocasa',base['text_model'])
    assert fh==load(Path(base['token_cache'])/'complete.json')['bundle_sha256']
    protocol=dict(state='APPROVED_GATE_2.5X',config_sha256=configs,
        revision='Opus 2026-10-12 11:50: retain p=.5; revise coverage gate 3x -> 2.5x before training; retain uniform coverage for late transport/lower state persistence',
        expectation='B tests more event exposure; closed-gripper never-lift share within lift targets changes 26.0% -> 32.3%, not a balanced conditional outcome prior. Dropout is a hypothesis, not a guaranteed improvement.',
        sampler_probability=.5,gate=2.5,actual_exposure_ratio=c['exposure_ratio'],
        schedule_sha256=pre['schedule_sha256'],selection_sha256=file_sha256(selection),
        feature_fingerprint=fh,implementation=implementation(),
        manifest_sha256={k:file_sha256(base[k]) for k in ('train_manifest','evaluation_manifest')},
        bundle_manifest_sha256=file_sha256(Path(base['bundle'])/'manifest.json'),
        source_sha256={str(ROOT/name):file_sha256(ROOT/name) for name in SCRIPTS},
        event_script_sha256=file_sha256(EXP/'gt_handoff_frozen.py'),
        metric_weight=dict(path='/media/tyshao/cache/torch/hub/checkpoints/vgg16-397923af.pth',
                           sha256=file_sha256('/media/tyshao/cache/torch/hub/checkpoints/vgg16-397923af.pth')),
        dropout_expected={g:pre['stats'][g]['dropout_fraction'] for g in ('all','success','failure')},
        A_run=str(ROOT/'runs/natural_sft_mixed_codec34000_10k_20261011'),
        A_events=str(ROOT/'evals_gt_handoff_codec34k_20261012'),
        event_eval=dict(labels=['start','+2'],beam=4,samples=8,selection_cases=51,bootstrap_unit='initial_state_group',bootstrap_repeats=10000,bootstrap_seed=20261012),
        frame_eval=dict(split='val',horizons=[10,20],beam=4,memory='full',models=['A','B','C','D'],
                        note='A fixed-horizon metrics are required for the LPIPS ratio; A event generation is not repeated'),
        safeguards=dict(success_lift_min=.95,partial_close_max='A',early_release_vanished_max='A',lpips20_max_ratio=1.05),
        scope='No test-set access, no P7/70%-failure/E4/E5/success-only training')
    frozen(EXP/'protocol.json',protocol)
    print('APPROVED_GATE_2.5X: configurations and unchanged p=.5 schedule frozen')

def preflight():
    protocol=load(EXP/'protocol.json');assert protocol['state']=='APPROVED_GATE_2.5X'
    for path,sha in protocol['source_sha256'].items():
        if file_sha256(path)!=sha:raise ValueError(f'Experiment code changed: {path}')
    assert implementation()==protocol['implementation']
    assert file_sha256(EXP/'gt_handoff_frozen.py')==protocol['event_script_sha256']
    assert file_sha256(EVAL/'selection.json')==protocol['selection_sha256']
    assert file_sha256(protocol['metric_weight']['path'])==protocol['metric_weight']['sha256']
    for label,sha in protocol['config_sha256'].items():assert file_sha256(EXP/f'{label}.json')==sha
    cfg=load(EXP/'B.json')
    assert file_sha256(cfg['schedule'])==protocol['schedule_sha256']
    assert feature_fingerprint(cfg['bundle'],'robocasa',cfg['text_model'])==protocol['feature_fingerprint']
    assert load(Path(cfg['token_cache'])/'complete.json')['bundle_sha256']==protocol['feature_fingerprint']
    for key,sha in protocol['manifest_sha256'].items():assert file_sha256(cfg[key])==sha
    assert file_sha256(Path(cfg['bundle'])/'manifest.json')==protocol['bundle_manifest_sha256']
    verify_bundle(Path(cfg['bundle']))
    assert file_sha256(Path(cfg['initial_checkpoint'])/'model.safetensors')==cfg['initial_sha256']
    train=read_manifest(cfg['train_manifest'],'robocasa');val=read_manifest(cfg['evaluation_manifest'],'robocasa')
    disjoint(train,val)
    for key in ('initial_state_id','initial_state_sha256'):
        assert not {r[key] for r in train['cases']} & {r[key] for r in val['cases']}
    # Validate every reusable cache entry, including checksums/source identity.
    from futureworlds.cache import load_cached
    for row in train['cases']:load_cached(cfg['token_cache'],row,protocol['feature_fingerprint'],'cpu')
    print('PASS: protocol, code, codec, RT1, split separation and all 800 cache entries')

def verify_checkpoint(folder,expected=None):
    receipt=load(folder/'experiment.json')
    if expected is not None:assert receipt['fingerprint']==expected
    for name,sha in receipt['files'].items():assert file_sha256(folder/name)==sha,folder/name
    return receipt

def runtime(label):
    cfg=load(EXP/f'{label}.json');out=Path(cfg['output']);fp=fingerprint(cfg)
    if (out/'run.json').exists():assert load(out/'run.json')['fingerprint']==fp
    if (out/'complete.json').exists():
        done=load(out/'complete.json');assert done['fingerprint']==fp and done['steps']==10000
        verify_checkpoint(out/'checkpoints/step_010000',fp)
        print('complete');return
    archive=EXP/'interrupted'/label/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    ckpts=sorted((out/'checkpoints').glob('step_*/experiment.json'))
    if ckpts:
        ckpt=ckpts[-1].parent;receipt=verify_checkpoint(ckpt,fp);step=receipt['step']
        cfg['resume']=str(ckpt)
        log=out/'metrics.jsonl'
        if log.exists():
            archive.mkdir(parents=True,exist_ok=True);shutil.copy2(log,archive/'metrics_before_resume.jsonl')
            kept=[]
            for line in log.read_text().splitlines():
                try:record=json.loads(line)
                except json.JSONDecodeError:continue
                if record['step']<=step:kept.append(record)
            assert [x['step'] for x in kept]==list(range(1,step+1))
            tmp=log.with_suffix('.resume.partial');tmp.write_text(''.join(json.dumps(x)+'\n' for x in kept));tmp.replace(log)
        for p in (out/'checkpoints').glob('*.partial'):
            archive.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(archive/p.name))
    elif out.exists() and any(out.iterdir()):
        # No published checkpoint: preserve interrupted run, restart from RT1/seed.
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out),str(archive))
    atomic(EXP/f'{label}_runtime.json',cfg);print('run')

def export_all():
    from futureworlds.exporting import export_model
    base=load(EXP/'B.json')
    for label in ('A','B','C','D'):
        run=Path(load(EXP/'protocol.json')['A_run']) if label=='A' else Path(load(EXP/f'{label}.json')['output'])
        done=load(run/'complete.json');assert done['state']=='COMPLETE' and done['steps']==10000
        cp=run/'checkpoints/step_010000';verify_checkpoint(cp)
        dest=EVAL/'bundles'/label;temp=dest.with_name(label+'.partial')
        if not dest.exists():
            if temp.exists():
                archive=EXP/'interrupted'/('export_'+label+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                archive.parent.mkdir(exist_ok=True);shutil.move(str(temp),str(archive))
            export_model(base['bundle'],'robocasa',cp,temp);temp.rename(dest)
        export_model(base['bundle'],'robocasa',cp,dest)
        print('Verified export',label,flush=True)

def verify_training():
    import math
    report={}
    for label in LABELS:
        cfg=load(EXP/f'{label}.json');out=Path(cfg['output']);fp=fingerprint(cfg)
        done=load(out/'complete.json');assert done['fingerprint']==fp and done['steps']==10000
        verify_checkpoint(out/'checkpoints/step_010000',fp)
        records=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
        assert [v['step'] for v in records]==list(range(1,10001))
        assert all(math.isfinite(v[k]) for v in records for k in ('loss','grad_norm','seconds'))
        item=dict(steps=len(records),fingerprint=fp)
        if label in ('C','D'):
            totals={k:sum(v['action_dropout_counts'][k] for v in records) for k in records[0]['action_dropout_counts']}
            pre=load(EXP/'preflight.json')['stats']
            for kind in ('success','failure'):
                assert totals[kind+'_samples']==pre[kind]['samples']
                assert totals[kind+'_dropped']==pre[kind]['dropout']
            item['dropout_counts']=totals
        report[label]=item
    atomic(EXP/'training_verification.json',report);print('All B/C/D final checkpoints and dropout counts verified')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','preflight','runtime','export','verify-training']);p.add_argument('--label',choices=list(LABELS))
    a=p.parse_args()
    if a.phase=='runtime':
        if a.label is None:p.error('--label required')
        runtime(a.label)
    else:{'freeze':freeze,'preflight':preflight,'export':export_all,'verify-training':verify_training}[a.phase]()
