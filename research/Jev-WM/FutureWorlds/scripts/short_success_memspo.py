"""New short, first-grasp-success data; original FutureWorlds MemSPO/R0."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from futureworlds.checkpoints import file_sha256,verify_bundle
from futureworlds.data import read_manifest,read_case
from futureworlds.sampling import Schedule
ROOT=Path('/media/tyshao/Jev-WM/data/short_success')
PROJECT=Path('/media/tyshao/Jev-WM/FutureWorlds')
OUT=PROJECT/'runs/short_success_memspo'
BASE=PROJECT/'assets/local_success_sft'
TEXT=Path('/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base')
STEPS=[50,100,150,200]
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def stable(p,v):
 if Path(p).exists() and read(p)!=v:raise ValueError(f'Existing recipe differs: {p}')
 write(p,v)
def prepare():
 audit=read(ROOT/'audit.json');assert audit['state']=='PASS' and audit['episodes']==300
 for split in ['train','validation','test']:
  path=ROOT/'packed'/split/'manifest.json'
  assert file_sha256(path)==audit['manifest_sha256'][split]
  m=read_manifest(path,'robocasa')
  assert len(m['cases'])==audit['counts'][split]
  for row in m['cases']:read_case(path,row,52,32)
 verify_bundle(BASE)
 cfg=dict(dataset='robocasa',method='memspo',bundle=str(BASE),initial_variant='sft',
  device='cuda',seed=20260918,steps=200,global_batch=16,lr=1e-6,adapter_lr=1e-4,
  warmup_steps=500,horizon=32,reward='R0',clip=.2,kl_coef=.001,diversity_penalty=.1,
  checkpoint_every=50,paper_locked=False,text_model=str(TEXT))
 cfg.update(train_manifest=str(ROOT/'packed/train/manifest.json'),
  evaluation_manifest=str(ROOT/'packed/validation/manifest.json'),
  output=str(OUT/'train'),schedule=str(OUT/'schedule.json'))
 cfg.pop('resume',None)
 assert cfg['method']=='memspo' and cfg['reward']=='R0' and cfg['horizon']==32
 assert cfg['steps']==200 and cfg['global_batch']==16 and cfg['bundle']==str(BASE)
 rows=read_manifest(cfg['train_manifest'],'robocasa')['cases']
 sampler=Schedule(rows,'robocasa',cfg['seed'])
 stable(OUT/'schedule.json',[dict(id=sampler.pick(i)[0]['id']) for i in range(cfg['steps']*cfg['global_batch'])])
 stable(OUT/'train.json',cfg)
 stable(OUT/'protocol.json',dict(dataset_protocol_sha256=audit['protocol_sha256'],dataset_manifests=audit['manifest_sha256'],
  initial_reference='original_episode_frames_0_and_4',horizon=32,
  selection='lowest validation mean LPIPS32 across steps50/100/150/200; ties earlier',
  validation_metric_horizons=[8,16,32],test_used_for_selection=False,
  algorithm='released MemSPO/R0; same settings as local_success_memspo_h32 except new dataset/output/schedule',
  base_bundle_manifest_sha256=file_sha256(BASE/'manifest.json'),
  limitations='New exact initial scenes; same simple counter task. Pixel metrics do not certify correct object behavior.'))
 print('Prepared: 200 train / 40 validation / 60 test, horizon32',flush=True)
def launch(command,args,gpus,distributed=True,dry=False):
 prefix=[sys.executable]
 if distributed:prefix+=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={gpus}']
 cmd=prefix+['-m','futureworlds',command]+list(map(str,args))
 print('RUN '+__import__('shlex').join(cmd),flush=True)
 if not dry:subprocess.run(cmd,check=True)
def evaluate(split,bundle,variant,dest,a):
 launch('evaluate',['--bundle',bundle,'--dataset','robocasa','--variant',variant,
  '--manifest',ROOT/'packed'/split/'manifest.json','--output',dest,'--device','cuda',
  '--metric-device','cuda','--text-model',TEXT,'--horizons',8,16,32,'--beam-width',4],a.gpus,dry=a.dry_run)
def train(a):
 cfg=read(OUT/'train.json');path=OUT/'train.json'
 if (OUT/'train/complete.json').exists():
  print('Training already complete',flush=True);return
 if (OUT/'train/run.json').exists():
  checkpoints=sorted((OUT/'train/checkpoints').glob('step_*/experiment.json'))
  if not checkpoints:raise RuntimeError('Existing unfinished run has no resumable checkpoint')
  cfg['resume']=str(checkpoints[-1].parent);path=OUT/'resume.json';write(path,cfg)
 launch('train',['--config',path],a.gpus,dry=a.dry_run)
def compare(a):
 candidates=[]
 for step in STEPS:
  checkpoint=OUT/'train/checkpoints'/f'step_{step:06d}'
  bundle=OUT/'validation_bundles'/f'step_{step:06d}'
  dest=OUT/'evaluation/validation'/f'step_{step:06d}'
  launch('export-model',['--bundle',BASE,'--dataset','robocasa','--checkpoint',checkpoint,'--output',bundle],a.gpus,False,a.dry_run)
  evaluate('validation',bundle,'main',dest,a)
  if not a.dry_run:
   summary=read(dest/'summary.json')
   assert summary['state']=='COMPLETE' and len(summary['per_case'])==40
   assert summary['protocol']['manifest_sha256']==file_sha256(ROOT/'packed/validation/manifest.json')
   candidates.append(dict(step=step,lpips32=summary['aggregate']['32']['lpips'],bundle=str(bundle)))
 if a.dry_run:
  print('Select only on validation, then evaluate baseline and selected RL on test.');return
 assert all(np.isfinite(c['lpips32']) for c in candidates)
 best=min(candidates,key=lambda c:(c['lpips32'],c['step']))
 stable(OUT/'checkpoint_selection.json',dict(candidates=candidates,selected=best,test_used_for_selection=False))
 evaluate('test',BASE,'sft',OUT/'evaluation/test/sft',a)
 evaluate('test',best['bundle'],'main',OUT/'evaluation/test/rl',a)
 report()
def report():
 selection=read(OUT/'checkpoint_selection.json');best=selection['selected']
 summaries={k:read(OUT/'evaluation/test'/k/'summary.json') for k in ['sft','rl']}
 assert all(s['state']=='COMPLETE' and len(s['per_case'])==60 for s in summaries.values())
 s={c['id']:c for c in summaries['sft']['per_case']}
 r={c['id']:c for c in summaries['rl']['per_case']};assert set(s)==set(r)
 lines=['# 新采集无抓取失败数据：MemSPO/R0 与 SFT 对比',
 f'\n训练200条，验证40条，测试60条；验证集选中第 {best["step"]} 步。',
 '\n| 预测帧数 | SFT LPIPS ↓ | RL LPIPS ↓ | 相对变化 |',
 '|---|---:|---:|---:|']
 result={}
 for h in ['8','16','32']:
  sv=summaries['sft']['aggregate'][h]['lpips'];rv=summaries['rl']['aggregate'][h]['lpips']
  delta=np.array([r[id]['metrics'][h]['lpips']-s[id]['metrics'][h]['lpips'] for id in sorted(s)])
  rng=np.random.default_rng(20261016)
  ci=np.quantile(delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1),[.025,.975]).tolist()
  result[h]=dict(sft=sv,rl=rv,relative_change_percent=100*(rv/sv-1),paired_delta_ci95=ci,improved_cases=int((delta<0).sum()))
  lines.append(f'| {h} | {sv:.6f} | {rv:.6f} | {100*(rv/sv-1):+.2f}% |')
 lines+=['\nLPIPS负变化为改善。须结合逐样本视频检查物体行为；指标改善不等于任务成功率提高。',
 f'\n32帧配对均值差95% bootstrap区间：{result["32"]["paired_delta_ci95"]}。',
 '\n本批为相同简化台面任务的新初态泛化；不是官方 RoboCasa benchmark，也不是新任务泛化。']
 write(OUT/'comparison.json',result);(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','baseline','train','compare','report','all'],nargs='?',default='all')
 p.add_argument('--gpus',type=int,choices=[4,8],default=4);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 if a.stage=='report':report();return
 prepare()
 if not a.dry_run and a.stage!='prepare':
  import torch
  if torch.cuda.device_count()<a.gpus:raise RuntimeError(f'Need {a.gpus} visible GPUs')
 if a.stage in ['baseline','all']:evaluate('validation',BASE,'sft',OUT/'evaluation/validation/sft',a)
 if a.stage in ['train','all']:train(a)
 if a.stage in ['compare','all']:compare(a)
if __name__=='__main__':main()

