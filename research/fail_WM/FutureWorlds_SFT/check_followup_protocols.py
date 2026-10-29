"""Targeted regression checks for endpoint ambiguity, invisibility and old-E1 correction."""
import json
from pathlib import Path
import numpy as np
from spatial_outcome_metrics import analyze, endpoint, point

def frame(x):
    image=np.zeros((256,320,3),np.uint8)
    image[98:103,198:203,1]=255
    if x is not None:image[98:103,x-2:x+3,0]=255
    return image

refs={'real':[100,100],'goal':[200,100],'start':[50,100]}
assert endpoint(None,refs)['class']=='undetected'
assert endpoint([195,100],refs)['class']=='goal_like'
assert endpoint([100,100],refs)['class']=='real_like'
assert endpoint([50,100],refs)['class']=='start_like'
assert endpoint([300,100],refs)['far_from_all']
assert endpoint([100,100],dict(refs,start=[100,100]))['ambiguous']
real=np.stack([frame(50),frame(50)]+[frame(100)]*6)
pred=np.stack([frame(100)]*5+[frame(None)])
result=analyze(pred,real)
assert result['class']=='undetected' and result['tail_detected']==4
assert result['last_missing'] and result['goalward_by_margin']['2'] is None
stationary=analyze(np.stack([frame(50)]*6),real)
assert stationary['stationary_prediction_proxy'] and stationary['class']=='start_like'
goalward=analyze(np.stack([frame(195)]*6),real)
assert goalward['goalward_by_margin']['2']
root=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
old=json.loads((root/'evals_e1_expanded_20261009/old_e1_spatial_audit.json').read_text())
b=next(r for r in old if r['model']=='natural' and r['id']=='g00006_b03' and r['name']=='beam4')
assert b['last_missing'] and b['class']=='undetected'
for cid,count in [('g00006_b03',2),('g00006_b04',4)]:
    rr=[r for r in old if r['model']=='natural' and r['id']==cid and r['name']!='beam4']
    assert sum(r['class']=='goal_like' for r in rr)==count
print('PASS: endpoint classes, overlap, outlier, final-frame disappearance, stationary proxy, goalward direction, and old E1 regression')
