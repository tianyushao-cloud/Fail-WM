"""Analyze completed matched two-step pilot and separate 20-step baseline."""
import json
from pathlib import Path
import numpy as np
from futureworlds.checkpoints import file_sha256
from futureworlds.data import write_json

ROOT=Path(__file__).resolve().parents[1]/'runs/jev_pilot_20261012'
arms=('sft','base_2step','hybrid_2step','base')
summaries={a:json.loads((ROOT/'evaluation'/a/'summary.json').read_text()) for a in arms}
frozen=json.loads((ROOT/'evaluation/frozen/summary.json').read_text())
ids=summaries['sft']['protocol']['cases']
for arm,s in summaries.items():
 assert s['state']=='COMPLETE' and s['protocol']['cases']==ids
 for c in s['per_case']:
  index=ids.index(c['id'])
  assert file_sha256(ROOT/'evaluation'/arm/f'case_{index:06d}'/'prediction.npz')==c['prediction_sha256']
rng=np.random.default_rng(20261012);draws=rng.integers(0,len(ids),(10000,len(ids)))
comparisons={}
for label,left,right in [('hybrid2_minus_base2','hybrid_2step','base_2step'),('base20_minus_sft','base','sft')]:
 comparisons[label]={}
 for h in ['4','8']:
  comparisons[label][h]={}
  for metric in ['psnr','ssim','lpips','mae','mse']:
   diff=np.array([x['metrics'][h][metric] for x in summaries[left]['per_case']])-np.array([x['metrics'][h][metric] for x in summaries[right]['per_case']])
   comparisons[label][h][metric]=dict(mean_difference=float(diff.mean()),paired_bootstrap_95=np.quantile(diff[draws].mean(1),[.025,.975]).tolist(),wins=int(np.sum(diff>0 if metric in ['psnr','ssim'] else diff<0)),n=len(ids))
motion={}
for arm in (*arms,'frozen'):
 rows=[]
 for i,cid in enumerate(ids):
  with np.load(ROOT/'evaluation'/('sft' if arm=='frozen' else arm)/f'case_{i:06d}'/'prediction.npz',allow_pickle=False) as z:
   truth=z['truth'].astype(np.float32)/255;previous=z['observations'][1:2].astype(np.float32)/255
   pred=np.repeat(previous,len(truth),axis=0) if arm=='frozen' else z['prediction']
  gt_delta=truth-np.concatenate([previous,truth[:-1]])
  delta=pred-np.concatenate([previous,pred[:-1]])
  mask=np.abs(truth-previous).mean(-1)>0.02
  roi_mae=float((np.abs(pred-truth)*mask[...,None]).sum()/max(1,3*mask.sum()))
  rows.append(dict(id=cid,delta_mae=float(np.abs(delta-gt_delta).mean()),predicted_motion=float(np.abs(delta).mean()),true_motion=float(np.abs(gt_delta).mean()),motion_roi_mae=roi_mae,roi_fraction=float(mask.mean())))
 motion[arm]=dict(per_case=rows,**{k:float(np.mean([r[k] for r in rows])) for k in ['delta_mae','predicted_motion','true_motion','motion_roi_mae','roi_fraction']})
training={}
for arm,steps in [('base',20),('base_2step',2),('hybrid_2step',2)]:
 ck=ROOT/arm/f'checkpoints/step_{steps:06d}';exp=json.loads((ck/'experiment.json').read_text())
 for name,digest in exp['files'].items():assert file_sha256(ck/name)==digest
 log=[json.loads(x) for x in (ROOT/arm/'metrics.jsonl').read_text().splitlines()]
 assert len(log)==steps and all(np.isfinite(r['grad_norm']) for r in log)
 training[arm]=dict(steps=steps,checkpoint=str(ck),core_sha256=file_sha256(ck/'model.safetensors'),resources=json.loads((ROOT/arm/'resources.json').read_text()))
# Replay must reproduce the first two completed live hybrid updates.
live=[json.loads(x) for x in (ROOT/'hybrid/metrics.jsonl').read_text().splitlines()]
replay=[json.loads(x) for x in (ROOT/'hybrid_2step/metrics.jsonl').read_text().splitlines()]
replay_differences=[dict(step=i+1,loss_abs=abs(a['loss']-b['loss']),grad_norm_abs=abs(a['grad_norm']-b['grad_norm']),reward_abs=abs(a['reward_metrics']['hybrid/mean']-b['reward_metrics']['hybrid/mean'])) for i,(a,b) in enumerate(zip(live,replay))]
cal=json.loads((ROOT/'calibration/summary.json').read_text())
receipts=[json.loads(p.read_text()) for p in (ROOT/'reviews').glob('*/score.json')]
tokens=sum(r['official_response']['usage']['input_tokens'] for r in receipts)+cal['input_tokens']
report=dict(state='PARTIAL_ORIGINAL_PLAN_COMPLETE_MATCHED_2STEP',aggregate={**{a:s['aggregate'] for a,s in summaries.items()},'frozen':frozen['aggregate']},
 comparisons=comparisons,motion=motion,training=training,replay_vs_live=replay_differences,calibration=cal,
 api=dict(successful_calibration_calls=24,successful_training_calls=len(receipts),input_tokens=tokens,base_cost_estimate_usd=tokens*.1/1e6,
          quota='50 RPD, used 50, official rate_limit_exceeded; 20-step hybrid blocked after 2 updates.'),
 limitations='Single seed, eight independent validation states, only two matched updates. No human preference labels or robot success measurement. Bootstrap unadjusted for multiple comparisons.')
write_json(ROOT/'comparison.json',report)
print(json.dumps(report,indent=2))
