import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT');OUT=ROOT/'diagnostics_drift_generalization_20261012'
def load(p):return json.loads(Path(p).read_text())
def summary(xs):
 xs=[x for x in xs if x is not None];return dict(n=len(xs),mean=float(np.mean(xs)),median=float(np.median(xs))) if xs else dict(n=0,mean=None,median=None)
def ci(xs):
 a=np.asarray(xs,float);rng=np.random.default_rng(20261012);m=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
 return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(m,[.025,.975]).tolist(),unit='matched_initial_state_stratum',seed=20261012,repeats=10000)
def main():
 protocol=load(OUT/'protocol.json');done=load(OUT/'complete.json');assert done['state']=='COMPLETE'
 results=[load(OUT/'cases'/c['split']/c['id']/'result.json') for c in protocol['cases']]
 rows=[];codec=[]
 for r in results:
  c=r['case'];L=c['L'];rest=r['rest'];thr=r['threshold']
  def extract(frames):
   v=[f for f in frames if f['t']>=L+2]
   return dict(states=dict(Counter(f['state'] for f in v)),endpoint=v[-1]['state'],
     high21_any=any(any(rest[1]-b['y']>thr for b in f['components']) for f in v),
     high21_frame_rate=float(np.mean([any(rest[1]-b['y']>thr for b in f['components']) for f in v])),
     none_frame_rate=float(np.mean([f['state']=='none' for f in v])),both_frame_rate=float(np.mean([f['state']=='both' for f in v])))
  codec.append(dict(id=c['id'],split=c['split'],role=c['role'],match=c['match'],gt=extract(r['gt']),codec=extract(r['codec'])))
  for mode,frames in r['metrics'].items():
   v=[f for f in frames if f['offset']>=2];assert [f['offset'] for f in v]==list(range(2,11))
   row=dict(id=c['id'],split=c['split'],role=c['role'],match=c['match'],mode=mode,**extract(v))
   row.update({k:summary([f[k] for f in v])['mean'] for k in ('nll','roi_nll','mae','roi_mae','outside_mae')})
   row['first_high21_offset']=next((f['offset'] for f in v if f['high21']),None)
   row['state_disagreement_with_gt']=float(np.mean([f['state']!=f['gt_state'] for f in v]))
   row['per_offset']=v;rows.append(row)
 aggregates=[]
 for split in ['train','val']:
  for role in ['failure','success']:
   for mode in ['oracle','late','start']:
    rr=[x for x in rows if x['split']==split and x['role']==role and x['mode']==mode]
    aggregates.append(dict(split=split,role=role,mode=mode,cases=len(rr),
      any_high21=sum(x['high21_any'] for x in rr),endpoints=dict(Counter(x['endpoint'] for x in rr)),
      **{k:summary([x[k] for x in rr]) for k in ['high21_frame_rate','none_frame_rate','both_frame_rate','nll','roi_nll','mae','roi_mae','outside_mae','state_disagreement_with_gt']}))
 by_failure_type=[]
 for kind,ids in [('skip_descend',[0,1,2]),('grasp_offset',[3,4,5])]:
  for split in ['train','val']:
   for mode in ['oracle','late','start']:
    rr=[x for x in rows if x['role']=='failure' and x['match'] in ids and x['split']==split and x['mode']==mode]
    by_failure_type.append(dict(type=kind,split=split,mode=mode,cases=len(rr),any_high21=sum(x['high21_any'] for x in rr),
       **{k:summary([x[k] for x in rr]) for k in ['high21_frame_rate','nll','roi_mae','none_frame_rate','both_frame_rate']}))
 index={(x['split'],x['role'],x['mode'],x['match']):x for x in rows}
 contrasts=[]
 for role in ['failure','success']:
  for metric in ['nll','roi_nll','roi_mae','mae','high21_frame_rate','high21_any','state_disagreement_with_gt']:
   for mode in ['oracle','late','start']:
    values=[float(index['val',role,mode,i][metric])-float(index['train',role,mode,i][metric]) for i in range(6)]
    contrasts.append(dict(name='val_minus_train',role=role,mode=mode,metric=metric,**ci(values)))
   for split in ['train','val']:
    for mode in ['late','start']:
     values=[float(index[split,role,mode,i][metric])-float(index[split,role,'oracle',i][metric]) for i in range(6)]
     contrasts.append(dict(name='autonomous_minus_oracle',split=split,role=role,mode=mode,metric=metric,**ci(values)))
   # Does unseen-state deficit get larger when autonomous histories replace GT?
   for mode in ['late','start']:
    values=[(float(index['val',role,mode,i][metric])-float(index['val',role,'oracle',i][metric]))-(float(index['train',role,mode,i][metric])-float(index['train',role,'oracle',i][metric])) for i in range(6)]
    contrasts.append(dict(name='val_minus_train_autonomous_penalty',role=role,mode=mode,metric=metric,**ci(values)))
 clean=[x for x in codec if x['role']=='failure' and not x['codec']['high21_any'] and x['codec']['none_frame_rate']==0]
 clean_keys={(x['split'],x['id']) for x in clean}
 clean_agg=[]
 for split in ['train','val']:
  for mode in ['oracle','late','start']:
   v=[x for x in rows if x['role']=='failure' and x['split']==split and x['mode']==mode and (x['split'],x['id']) in clean_keys]
   clean_agg.append(dict(split=split,mode=mode,cases=len(v),any_high21=sum(x['high21_any'] for x in v),
                         high21_frame_rate=summary([x['high21_frame_rate'] for x in v]),nll=summary([x['nll'] for x in v]),roi_mae=summary([x['roi_mae'] for x in v])))
 trajectories=[]
 for split in ['train','val']:
  for role in ['failure','success']:
   for mode in ['oracle','late','start']:
    rr=[x for x in rows if x['split']==split and x['role']==role and x['mode']==mode]
    for offset in range(2,11):
     vs=[x['per_offset'][offset-2] for x in rr]
     trajectories.append(dict(split=split,role=role,mode=mode,offset=offset,
       high21=sum(v['high21'] for v in vs),states=dict(Counter(v['state'] for v in vs)),
       nll=summary([v['nll'] for v in vs])['mean'],roi_mae=summary([v['roi_mae'] for v in vs])['mean']))
 result=dict(state='COMPLETE',aggregates=aggregates,by_failure_type=by_failure_type,paired_contrasts=contrasts,codec=codec,
             clean_failure_subset=clean_agg,case_rows=rows,by_offset=trajectories,
             interpretation_limits=protocol['limitations']+['Original threshold applied to any red connected component; differs from older centroid-of-all-red lift detector, so these counts are not interchangeable with 2x2 report.',
               'Oracle means GT previous frames with within-frame free beam generation, not target-token teacher forcing. NLL uses separate teacher-forced scoring of that same target.'])
 (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 lines=['# 区分真实历史条件误差与自主生成误差','','24例，train/val各6失败+6成功。主窗口统一L+2..L+10。每例只有一条beam4输出，不混入采样。',
 'oracle：每一帧都给真实历史再生成下一帧；late：L+2之前给真实历史，此后自由生成；start：只有最初2帧真实图像。','',
 '|split|类别|条件|窗口曾抬升|高处帧占比|方块ROI MAE|全帧MAE|真实target NLL|末帧状态|',
 '|---|---|---|---:|---:|---:|---:|---:|---|']
 for a in aggregates:
  lines.append(f"|{a['split']}|{a['role']}|{a['mode']}|{a['any_high21']}/{a['cases']}|{a['high21_frame_rate']['mean']:.1%}|{a['roi_mae']['mean']:.2f}|{a['mae']['mean']:.2f}|{a['nll']['mean']:.3f}|{a['endpoints']}|")
 lines+=['','注意：成功案例抬升是正确行为，失败案例抬升是错误。高处是红连通块代理，不是检测到了真实夹爪。both/none/other保留。','',
         '## 关键差值与区间','','按6个匹配初态层bootstrap 10000次，仅探索性诊断。','',
         '|比较|类别|条件|指标|均值差|95%区间|','|---|---|---|---|---:|---|']
 for x in contrasts:
  if x['role']=='failure' and x['metric'] in ['nll','high21_frame_rate','roi_mae'] and (x['name']=='val_minus_train' and x['mode']=='oracle' or x['name']=='autonomous_minus_oracle'):
   name=x['name']+('/'+x['split'] if 'split'in x else '')
   lines.append(f"|{name}|{x['role']}|{x['mode']}|{x['metric']}|{x['mean']:.4f}|[{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]|")
 lines+=['','## 限制','']+['- '+x for x in result['interpretation_limits']]+['','逐例证据：cases/{train,val}/{id}/comparison.jpg，列为GT/codec/oracle/late/start；每行一个目标帧。完整生成token和浮点图像保留于各mode.npz。']
 (OUT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
