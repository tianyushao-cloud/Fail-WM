"""Post-hoc diagnostic only: uncertainty bounds, never filled-in training rewards."""
from pathlib import Path
import itertools,json
ROOT=Path(__file__).resolve().parent
rs=[json.loads(p.read_text()) for p in (ROOT/'requests').glob('*/receipt.json')]
lookup={(r['identity']['case']['id'],r['identity']['order'],r['identity']['repeat']):r for r in rs}
dims=['structure','dynamics','interaction']
def value(labels):
    values={'wrong':0.,'partial':.5,'correct':1.}
    v=sum(w*values[l] for w,l in zip([.2,.4,.4],labels))
    if labels[0]=='wrong':v=min(v,.25)
    if labels[2]=='wrong':v=min(v,.5)
    return v
real_agreement=[];stable_cells=[];bounds=[]
for scene in range(1,4):
    for kind in ['sft_full','sft_recap','rl_recap']:
        key=f'scene{scene}_{kind}';a=lookup[key,'reference_first',0];b=lookup[key,'prediction_first',0]
        r=lookup.get((key,'reference_first',1));review=[a,b]+([r] if r else [])
        segments=[]
        for s in range(1,5):
            choices=[];labels={}
            for d in dims:
                name=f's{s}_{d}'
                real_agreement.append(a['answers'][name]['raw']==b['answers'][name]['raw'])
                accepted=[x['answers'][name]['accepted'] for x in review]
                stable=len(set(accepted))==1 and accepted[0]!='uncertain'
                stable_cells.append(stable)
                labels[d]=accepted[0] if stable else 'uncertain'
                choices.append([accepted[0]] if stable else ['wrong','partial','correct'])
            vals=[value(x) for x in itertools.product(*choices)]
            segments.append(dict(segment=s,labels=labels,lower=min(vals),upper=max(vals)))
        bounds.append(dict(id=key,scene=scene,kind=kind,lower=sum(x['lower'] for x in segments)/4,upper=sum(x['upper'] for x in segments)/4,segments=segments))
pairs=[]
for scene in range(1,4):
    ss={x['kind']:x for x in bounds if x['scene']==scene}
    for other in ['sft_full','sft_recap']:
        a,b=ss['rl_recap'],ss[other]
        pairs.append(dict(scene=scene,A='rl_recap',B=other,A_lower=a['lower'],A_upper=a['upper'],B_lower=b['lower'],B_upper=b['upper'],strict_preference='A' if a['lower']>b['upper']+1e-9 else 'B' if b['lower']>a['upper']+1e-9 else 'undetermined'))
out=dict(post_hoc=True,description='Not a statistical confidence interval. Enumerates all correct/partial/wrong assignments for dimensions not confidently agreed across presentation orders and available repeats. Keeps original weights/caps. No request thresholds changed. No optimizer use.',real_order_agreement=dict(numerator=sum(real_agreement),denominator=len(real_agreement),fraction=sum(real_agreement)/len(real_agreement)),stable_real_cells=dict(numerator=sum(stable_cells),denominator=len(stable_cells),fraction=sum(stable_cells)/len(stable_cells)),bounds=bounds,pairs=pairs)
(ROOT/'uncertainty_analysis.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
