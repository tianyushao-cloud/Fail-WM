import json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
R=Path('/media/tyshao/Jev-WM/FutureWorlds/runs/local_success_memspo')
D=R/'diagnosis/final_review';D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
out={'splits':{},'videos':[]}
p=read(R/'protocol.json')
assert p['initial_reference_policy']=='original_episode_first_two'
metrics=[json.loads(l) for l in (R/'train/metrics.jsonl').read_text().splitlines() if l.strip()]
assert [r['step'] for r in metrics]==list(range(1,201))
assert all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in metrics)
out['training']=dict(steps=200,sequential_steps=True,finite_metrics=True,train_minutes=sum(x['seconds'] for x in metrics)/60,
 first_loss=metrics[0]['loss'],last_loss=metrics[-1]['loss'])
sheets=[]
for split in ['validation','test']:
 m=R/f'data/{split}/manifest.json';rows=read(m)['cases']
 summaries={a:read(R/f'evaluation/{split}/{a}/summary.json') for a in ['sft','rl']}
 for k in ['manifest_sha256','cases','horizons','beam_width','memory','text_assets','implementation','precision','gt_refresh']:
  assert summaries['sft']['protocol'][k]==summaries['rl']['protocol'][k],k
 assert summaries['sft']['protocol']['manifest_sha256']==sha(m)
 records=[]
 for i,row in enumerate(rows):
  assert row['source_start']==0 and row['success'] is True
  arrays={}
  for arm in ['sft','rl']:
   folder=R/f'evaluation/{split}/{arm}/case_{i:06d}';rec=read(folder/'result.json')
   assert rec['id']==row['id'] and rec['prediction_sha256']==sha(folder/'prediction.npz')
   assert rec['fingerprint']==summaries[arm]['fingerprint']
   with np.load(folder/'prediction.npz') as z:arrays[arm]={k:z[k].copy() for k in z.files}
   assert arrays[arm]['prediction'].shape==(32,256,320,3)
   assert np.isfinite(arrays[arm]['prediction']).all()
  s,r=arrays['sft'],arrays['rl']
  assert np.array_equal(s['truth'],r['truth']) and np.array_equal(s['observations'],r['observations'])
  ss=summaries['sft']['per_case'][i];rr=summaries['rl']['per_case'][i]
  record=dict(id=row['id'],episode_id=row['episode_id'],group=row['initial_state_id'],index=i,
   sft_lpips=ss['metrics']['32']['lpips'],rl_lpips=rr['metrics']['32']['lpips'],
   delta=rr['metrics']['32']['lpips']-ss['metrics']['32']['lpips'],
   movement_gt=float(np.abs(np.diff(s['truth'].astype(np.float32)/255,axis=0)).mean()),
   movement_sft=float(np.abs(np.diff(s['prediction'],axis=0)).mean()),
   movement_rl=float(np.abs(np.diff(r['prediction'],axis=0)).mean()),
   token_change_sft=float(np.mean(s['tokens'][:,1:]!=s['tokens'][:,:-1])),
   token_change_rl=float(np.mean(r['tokens'][:,1:]!=r['tokens'][:,:-1])),
   late_lpips_sft=float(np.mean(ss['per_frame']['lpips'][20:32])),
   late_lpips_rl=float(np.mean(rr['per_frame']['lpips'][20:32])))
  records.append(record)
  if (i+1)%20==0:print(split,i+1,'audited',flush=True)
 grouped={}
 for group in sorted({x['group'] for x in records}):
  x=[r for r in records if r['group']==group];grouped[group]=float(np.mean([r['delta'] for r in x]))
 ordered=sorted(records,key=lambda r:r['delta'])
 out['splits'][split]=dict(cases=len(records),all_paired_hashes_and_truth_pass=True,
  improved_cases=sum(r['delta']<0 for r in records),improved_initial_groups=sum(x<0 for x in grouped.values()),
  group_delta=grouped,severe_regressions_over_0_05=sum(r['delta']>.05 for r in records),
  median_movement={k:float(np.median([r['movement_'+k] for r in records])) for k in ['gt','sft','rl']},
  mean_token_change={k:float(np.mean([r['token_change_'+k] for r in records])) for k in ['sft','rl']},
  late_lpips={k:float(np.mean([r['late_lpips_'+k] for r in records])) for k in ['sft','rl']},records=records)
 for label,rec in [('most_improved',ordered[0]),('median_delta',ordered[len(ordered)//2]),('most_regressed',ordered[-1])]:
  i=rec['index'];movies=[]
  with np.load(R/f'evaluation/{split}/sft/case_{i:06d}/prediction.npz') as z:
   obs=z['observations'].copy();gt=np.concatenate([obs,z['truth']],axis=0)
   s=np.concatenate([obs,np.rint(z['prediction']*255).clip(0,255).astype(np.uint8)],axis=0)
  with np.load(R/f'evaluation/{split}/rl/case_{i:06d}/prediction.npz') as z:
   r=np.concatenate([obs,np.rint(z['prediction']*255).clip(0,255).astype(np.uint8)],axis=0)
  for t in range(34):
   im=Image.new('RGB',(960,296),'black');draw=ImageDraw.Draw(im)
   for k,(mov,title) in enumerate(zip([gt,s,r],['GT','SFT','RL step200'])):
    im.paste(Image.fromarray(mov[t]),(320*k,40));draw.text((320*k+4,3),title,fill='white')
    draw.text((320*k+4,19),f"{split} {rec['episode_id']} t={t}",fill='yellow')
   movies.append(np.asarray(im))
  path=D/f"{split}_{label}_{rec['episode_id']}_GT_SFT_RL.mp4"
  subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x296','-r','5','-i','pipe:0','-an','-c:v','libx264','-crf','14','-pix_fmt','yuv420p',str(path)],input=np.stack(movies).tobytes(),capture_output=True,check=True)
  out['videos'].append(dict(split=split,label=label,case=rec['episode_id'],delta=rec['delta'],path=str(path)))
  sheet=Image.new('RGB',(960,296*3))
  for j,t in enumerate([11,21,33]):sheet.paste(Image.fromarray(movies[t]),(0,296*j))
  sheet.save(path.with_suffix('.jpg'),quality=90)
  sheets.append(movies[-1])
 print(split,'complete',flush=True)
sheet=Image.new('RGB',(960,296*len(sheets)))
for j,frame in enumerate(sheets):sheet.paste(Image.fromarray(frame),(0,296*j))
sheet.save(D/'final_frames.jpg',quality=90)
(D/'audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:({s:{a:b for a,b in v.items() if a!='records'} for s,v in x.items()} if k=='splits' else x) for k,x in out.items()},ensure_ascii=False,indent=2),flush=True)

