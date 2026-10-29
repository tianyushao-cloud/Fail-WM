import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
BASE=ROOT.parent/'rt1_memspo200_upstream_20261017'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','all'],default='all',nargs='?');args=parser.parse_args()
    cfg=read(ROOT/'train.json');baseline=read(BASE/'train.json')
    for k in ['dataset','method','initial_variant','steps','global_batch','lr','adapter_lr','warmup_steps','horizon','clip','kl_coef','diversity_penalty','checkpoint_every','seed']:
        assert cfg[k]==baseline[k],k
    assert cfg['reward']=='jev_gt_pair' and cfg['decisions']['backend']=='queue'
    for name,expected in read(ROOT/'source_hashes.json').items():
        assert digest(SOURCE/name)==expected,name
    original=read(ROOT/'upstream_source_hashes.json')
    changed=[name for name,h in original.items() if digest(SOURCE/name)!=h]
    assert changed==['src/futureworlds/training.py'],changed
    assert digest(cfg['schedule'])==digest(baseline['schedule'])
    sys.path.insert(0,str(SOURCE/'src'))
    from futureworlds.data import read_manifest,disjoint
    train=read_manifest(cfg['train_manifest'],'rt1')
    val=read_manifest(ROOT/'data/validation/manifest.json','rt1')
    test=read_manifest(cfg['evaluation_manifest'],'rt1')
    disjoint(train,val);disjoint(train,test);disjoint(val,test)
    schedule=read(cfg['schedule']);assert len(schedule)==3200
    assert {x['id'] for x in schedule}<={x['id'] for x in train['cases']}
    from futureworlds.checkpoints import verify_bundle
    verify_bundle(cfg['bundle'])
    info=dict(gpus=4,accumulation_per_gpu=4,scheduled_batches=200,global_batch=16,horizon=10,
              reward='pure GT-conditioned Jev; no R0 reward or R0 selection',initial_variant='sft',
              train_episodes=len(train['cases']),validation_episodes=len(val['cases']),test_episodes=len(test['cases']),
              original_hyperparameters_match=True,source_verified=True,
              caveat='200 scheduled batches; all-tied batches skip optimizer and are counted separately')
    print(json.dumps(info,indent=2),flush=True);write(ROOT/'preparation_check.json',info)
    if args.stage=='check':return
    import torch
    if torch.cuda.device_count()<4:raise RuntimeError(f'Need four visible GPUs, found {torch.cuda.device_count()}; no fallback')
    control=read(ROOT/'api_control_check.json')
    if control['state']!='PASS':raise RuntimeError('API control check not passed')
    broker=read(Path(cfg['decisions']['review_dir'])/'broker.json')
    if broker['state']!='RUNNING' or time.time()-broker['heartbeat_unix']>60:
        raise RuntimeError('Judge service unavailable; start it on gateway with start_judge.sh')
    write(ROOT/'launch.json',dict(**info,pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    env={**os.environ,'PYTHONPATH':str(SOURCE/'src'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'USE_TF':'0','USE_FLAX':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
         'TOKENIZERS_PARALLELISM':'false'}
    def command(stage,params,distributed=True):
        prefix=[sys.executable,'-u']
        if distributed:prefix+=['-m','torch.distributed.run','--standalone','--nproc_per_node=4']
        cmd=prefix+['-m','futureworlds']+[str(x) for x in params]
        write(ROOT/'status.json',dict(stage=stage,command=cmd,pid=os.getpid()))
        print('STAGE',stage,flush=True)
        with (ROOT/'logs'/f'{stage}.log').open('a') as stream:
            subprocess.run(cmd,cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    command('cache',['cache','--bundle',cfg['bundle'],'--dataset','rt1','--manifest',cfg['train_manifest'],
                     '--output',cfg['token_cache'],'--device','cuda','--text-model',cfg['text_model']])
    def training(stage,config):
        c=read(config);out=Path(c['output'])
        if (out/'complete.json').exists():return
        checkpoints=sorted((out/'checkpoints').glob('step_*/experiment.json'))
        if checkpoints:
            c['resume']=str(checkpoints[-1].parent);config=ROOT/f'{stage}_resume.json';write(config,c)
        elif (out/'run.json').exists():
            # No optimizer state has been persisted; restart from the same SFT,
            # retaining failed attempt evidence and exact content-keyed receipts.
            archive=out.with_name(out.name+'.incomplete.'+str(time.time_ns()))
            out.rename(archive)
        command(stage,['train','--config',config])
    training('distributed_smoke',ROOT/'smoke.json')
    smoke=[json.loads(x) for x in (ROOT/'smoke/train/metrics.jsonl').read_text().splitlines()]
    if not any(x['optimizer_updated'] for x in smoke):
        raise RuntimeError('Distributed smoke produced no informative preferences; formal training not started')
    training('train',ROOT/'train.json')
    command('export',['export-model','--bundle',cfg['bundle'],'--dataset','rt1','--checkpoint',
                      ROOT/'train/checkpoints/step_000200','--output',ROOT/'rl_bundle'],distributed=False)
    for split in ['validation','test']:
        for label,bundle,variant in [('sft',cfg['bundle'],'sft'),('rl',ROOT/'rl_bundle','main')]:
            command(f'evaluate_{split}_{label}',['evaluate','--bundle',bundle,'--dataset','rt1','--variant',variant,
                    '--manifest',ROOT/f'data/{split}/manifest.json','--output',ROOT/f'evaluation/{split}/{label}',
                    '--device','cuda','--text-model',cfg['text_model'],'--horizons',10,20,32,'--beam-width',4])
    subprocess.run([sys.executable,str(ROOT/'report.py')],cwd=ROOT,env=env,check=True)
    write(ROOT/'status.json',dict(stage='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__=='__main__':
    lock=(ROOT/'.launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except Exception as error:
        write(ROOT/'status.json',dict(stage='FAILED',error=repr(error)))
        raise
