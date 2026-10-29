import json,collections,re
from pathlib import Path
import numpy as np
root=Path('/media/tyshao/Jev-WM/FutureWorlds/runs/local_success_memspo_h32')
out=root/'diagnosis';out.mkdir(exist_ok=True)
load=lambda p:json.loads(p.read_text())
base=load(root/'evaluation/validation/sft/summary.json')
models={'sft':base,**{str(s):load(root/f'validation_checkpoints/step_{s:06d}/summary.json') for s in [50,100,150,200]}}
indices={k:{r['id']:r for r in v['per_case']} for k,v in models.items()}
sourcecache={};meta={}
for split in ['train','validation']:
 for row in load(root/f'data/{split}/manifest.json')['cases']:
  src=row['source_manifest']
  if src not in sourcecache:sourcecache[src]={r['id']:r for r in load(Path(src))['cases']}
  sr=sourcecache[src][row['episode_id']]
  m=load(Path(sr['source_h5']).parent/'metadata.json')
  meta[row['id']]=dict(category=m['category'],failure=any(e['event']=='grasp_attempt_failed' for e in m.get('events',[])),group=row['initial_state_id'])
ids=list(indices['sft']);groups=sorted({meta[i]['group'] for i in ids})
def comparison(selected,left,right,start=0,end=32):
 d=np.array([np.mean(indices[right][i]['per_frame']['lpips'][start:end])-np.mean(indices[left][i]['per_frame']['lpips'][start:end]) for i in selected])
 x=np.mean([np.mean(indices[left][i]['per_frame']['lpips'][start:end]) for i in selected])
 y=np.mean([np.mean(indices[right][i]['per_frame']['lpips'][start:end]) for i in selected])
 return dict(n=len(d),left=float(x),right=float(y),delta=float(d.mean()),relative_percent=float((y/x-1)*100),improved=int((d<0).sum()))
report={'checkpoint_curve':{k:v['aggregate']['32'] for k,v in models.items()},'subgroups':{},'time_segments':{},'group_changes':[]}
for name,subset in [('all',ids),('prior_failure',[i for i in ids if meta[i]['failure']]),('no_recorded_failure',[i for i in ids if not meta[i]['failure']])]+[(c,[i for i in ids if meta[i]['category']==c]) for c in sorted({meta[i]['category'] for i in ids})]:
 report['subgroups'][name]={f'{l}_to_{r}':comparison(subset,l,r) for l,r in [('sft','100'),('sft','200'),('100','200')]}
for name,start,end in [('first10',0,10),('middle10',10,20),('last12',20,32)]:
 report['time_segments'][name]={f'{l}_to_{r}':comparison(ids,l,r,start,end) for l,r in [('sft','100'),('sft','200'),('100','200')]}
for group in groups:
 sub=[i for i in ids if meta[i]['group']==group]
 report['group_changes'].append(dict(group=group,**comparison(sub,'100','200')))
rng=np.random.default_rng(20261016)
samples=[]
for _ in range(10000):
 selected=[i for g in rng.choice(groups,len(groups),replace=True) for i in ids if meta[i]['group']==g]
 samples.append(np.mean([indices['200'][i]['metrics']['32']['lpips']-indices['100'][i]['metrics']['32']['lpips'] for i in selected]))
report['step200_minus100_cluster_bootstrap95']=np.quantile(samples,[.025,.975]).tolist()
rank=sorted(ids,key=lambda i:indices['200'][i]['metrics']['32']['lpips']-indices['100'][i]['metrics']['32']['lpips'],reverse=True)
report['worst_step200_vs100']=[dict(id=i,**meta[i],**comparison([i],'100','200')) for i in rank[:8]]
sched=load(root/'schedule.json');report['schedule']={}
for lo,hi in [(0,100),(100,200),(0,200)]:
 rows=sched[lo*16:hi*16];counts=collections.Counter(r['id'] for r in rows)
 report['schedule'][f'{lo+1}-{hi}']=dict(exposures=len(rows),unique_episodes=len(counts),unique_groups=len({meta[i]['group'] for i in counts}),mean_repeats=len(rows)/len(counts),categories=dict(collections.Counter(meta[r['id']]['category'] for r in rows)),prior_failure_fraction=sum(meta[r['id']]['failure'] for r in rows)/len(rows))
# Parse each rank's scalar prints without claiming they are fixed-sample reward curves.
chunks=[];vals=[]
for line in (root/'logs/all.log').read_text().splitlines():
 match=re.fullmatch(r'loss ([-+0-9.eE]+)',line)
 if match:vals.append(float(match.group(1)))
 if line.startswith('{"step":'):
  r=json.loads(line);chunks.append(dict(step=r['step'],values=vals));vals=[]
report['logged_candidate_error']={}
for lo,hi in [(1,50),(51,100),(101,150),(151,200)]:
 values=[v for r in chunks if lo<=r['step']<=hi for v in r['values']]
 report['logged_candidate_error'][f'{lo}-{hi}']=dict(count=len(values),mean=float(np.mean(values)),median=float(np.median(values)))
(out/'step200_analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

