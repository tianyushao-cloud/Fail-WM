"""Exploratory soft-grade analysis; does not replace preregistered v2 gates."""
from pathlib import Path
import itertools,json,time
ROOT=Path(__file__).resolve().parent
RULE=dict(version='soft_grade_exploration_v1',label_values=dict(correct=1,partial=.5,wrong=0),uncertain='Unknown contribution ranges from0 to1, never filled with0 as a reward.',method='Expected ordinal grade from categorical probabilities; propagate uncertain mass as interval. Across layouts/repeats take the envelope. Apply per-segment cap only for unanimously confidently severe structural/interaction error. Require interval separation >0.05 for an exploratory pair preference.',minimum_pair_gap=.05,not_calibrated=True,optimizer_use=False,original_v2_gates_unchanged=True)
if not (ROOT/'soft_rule.json').exists():(ROOT/'soft_rule.json').write_text(json.dumps(dict(**RULE,declared_unix=time.time()),indent=2))
receipts=[json.loads(p.read_text()) for p in (ROOT/'requests').glob('*/receipt.json')]
lookup={(r['identity']['case']['id'],r['identity']['segment'],r['identity']['layout'],r['identity']['repeat']):r for r in receipts}
cases=json.loads((ROOT/'cases.json').read_text());records=[]
for c in cases:
    all_segments=[]
    for s in range(4):
        rr=[lookup.get((c['id'],s,l,r)) for l,r in [('reference_left',0),('prediction_left',0),('reference_left',1)]]
        if not rr[0] or not rr[1]:break
        rr=[r for r in rr if r];cap=1.
        for d,limit in [('structure',.25),('interaction',.5)]:
            if all(r['answers'][f's{s+1}_{d}']['accepted']=='wrong' for r in rr):cap=min(cap,limit)
        intervals=[]
        for r in rr:
            lower=upper=0.
            for d,w in [('structure',.2),('dynamics',.4),('interaction',.4)]:
                a=next(a for a in r['response']['answers'] if a['name']==f's{s+1}_{d}')
                probabilities={p['value']:p['probability'] for p in a['probabilities']};total=sum(probabilities.values())
                # API probabilities are rounded; normalize them before aggregation.
                probabilities={k:v/total for k,v in probabilities.items()}
                e=probabilities['correct']+.5*probabilities['partial'];u=probabilities['uncertain']
                lower+=w*e;upper+=w*(e+u)
            intervals.append(dict(layout=r['identity']['layout'],repeat=r['identity']['repeat'],lower=min(lower,cap),upper=min(upper,cap)))
        all_segments.append(dict(segment=s+1,lower=min(i['lower'] for i in intervals),upper=max(i['upper'] for i in intervals),layouts=intervals))
    if len(all_segments)!=4:continue
    records.append(dict(id=c['id'],kind=c['kind'],scene=c['scene'],cohort=c['cohort'],lower=sum(s['lower'] for s in all_segments)/4,upper=sum(s['upper'] for s in all_segments)/4,segments=all_segments))
pairs=[]
for scene in [4,5]:
    candidates=[r for r in records if r['scene']==scene and r['kind'].startswith('rl_g')]
    for a,b in itertools.combinations(candidates,2):
        winner=a['id'] if a['lower']-b['upper']>.05 else b['id'] if b['lower']-a['upper']>.05 else None
        pairs.append(dict(scene=scene,A=a['id'],B=b['id'],winner=winner))
out=dict(rule=RULE,completed_requests=len(receipts),records=records,validation_pairs=pairs,informative_pairs=sum(p['winner'] is not None for p in pairs))
(ROOT/'soft_diagnostic.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(completed_requests=len(receipts),records=len(records),informative_pairs=out['informative_pairs'])))
