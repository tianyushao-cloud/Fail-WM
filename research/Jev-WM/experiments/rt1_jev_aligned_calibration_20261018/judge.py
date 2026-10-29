"""Per-segment aligned evidence; same categorical rubric/thresholds as v1."""
from pathlib import Path
import argparse,base64,fcntl,hashlib,io,itertools,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'rt1_jev_segment_calibration_20261018'
sys.path.insert(0,str(OLD))
import calibrate as previous
from runtime import api_environment
from futureworlds.decision_api import DecisionsClient
from futureworlds.jev_judge import atomic_json,canonical,sha

PROMPT=previous.PROMPT.replace('All 32 future frames are provided.','All 32 future frames appear in the OVERVIEW contact sheet, read row by row from left to right. Each tile pairs REFERENCE and PREDICTION at the same timestamp. Separate full-resolution paired images provide every one of the 8 TARGET segment frames, plus the preceding frame where available. Score ONLY the TARGET segment. The overview is context, not another segment to grade. Explicit labels identify roles; their left/right positions may change. Ignore panel borders, text labels and layout when assessing image quality.')
MODEL=previous.MODEL
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)

def paired(ref,pred,index,layout,small=False):
    size=(160,128) if small else (320,256)
    out=Image.new('RGB',(size[0]*2,size[1]+28),(18,23,32));d=ImageDraw.Draw(out)
    entries=[('REF',ref),('PRED',pred)]
    if layout=='prediction_left':entries.reverse()
    for col,(name,frame) in enumerate(entries):
        im=Image.fromarray(frame)
        if small:im=im.resize(size,Image.Resampling.LANCZOS)
        out.paste(im,(col*size[0],28));d.text((col*size[0]+4,4),f'{name} future {index}',font=FONT,fill='white')
    return out

def evidence(arrays,segment,layout):
    images=[]
    initial=Image.new('RGB',(640,284),(18,23,32));d=ImageDraw.Draw(initial)
    for i,frame in enumerate(arrays['initial']):initial.paste(Image.fromarray(frame),(i*320,28));d.text((i*320+4,4),f'SHARED INITIAL {i+1}/2',font=FONT,fill='white')
    images.append(('Shared two initial observations',initial))
    overview=Image.new('RGB',(1280,1280),(18,23,32));d=ImageDraw.Draw(overview)
    d.text((8,4),'OVERVIEW: 32 future timestamps, left-to-right then next row. TARGET frames '+str(segment*8+1)+'-'+str(segment*8+8),font=FONT,fill='white')
    for t in range(32):overview.paste(paired(arrays['reference'][t],arrays['prediction'][t],t+1,layout,True),((t%4)*320,32+(t//4)*156))
    images.append(('OVERVIEW ONLY: complete 32-frame paired trajectory, row-major order',overview))
    start=segment*8
    if start:images.append((f'BOUNDARY CONTEXT ONLY, future frame {start}',paired(arrays['reference'][start-1],arrays['prediction'][start-1],start,layout)))
    for t in range(start,start+8):images.append((f'TARGET future frame {t+1}, segment {segment+1}',paired(arrays['reference'][t],arrays['prediction'][t],t+1,layout)))
    return images

def questions(segment):
    qs=previous.questions()
    return [q for q in qs if q['name'].startswith(f's{segment+1}_')]

def reserve():
    with (ROOT/'.budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);p=ROOT/'http_budget.json'
        x=json.loads(p.read_text()) if p.exists() else dict(attempts=0)
        if x['attempts']>=220:raise RuntimeError('Shared HTTP attempt cap 220 reached')
        x['attempts']+=1;atomic_json(p,x)

def work(meta,segment,layout,repeat):
    identity=dict(case=meta,segment=segment,layout=layout,repeat=repeat,prompt=PROMPT,questions=questions(segment),model=MODEL,render_version='paired_fullres_target8_all32_overview_v1')
    key=sha(canonical(identity).encode());folder=ROOT/'requests'/key;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json'
    if receipt.exists():return json.loads(receipt.read_text())
    with np.load(ROOT/'cases'/meta['id']/'frames.npz') as z:arrays={k:z[k].copy() for k in z.files}
    content=[dict(type='input_text',text=PROMPT+'\nTARGET: future frames '+str(segment*8+1)+' through '+str(segment*8+8)+'.\nTASK DATA: '+json.dumps(meta['task']))]
    descriptors=[]
    for i,(label,im) in enumerate(evidence(arrays,segment,layout)):
        b=io.BytesIO();im.save(b,format='PNG');data=b.getvalue()
        descriptors.append(dict(label=label,size=list(im.size),sha256=sha(data)))
        content.extend([dict(type='input_text',text=label),dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(data).decode())])
        if meta['id'] in ['scene1_rl_recap','val1_freeze','val1_rl_g0'] and segment==2 and repeat==0:
            f=ROOT/'input_examples'/f'{meta["id"]}_{layout}';f.mkdir(parents=True,exist_ok=True);(f/f'{i:02d}.png').write_bytes(data)
    atomic_json(folder/'identity.json',identity);atomic_json(folder/'image_manifest.json',descriptors)
    payload=dict(model=MODEL,input=[dict(role='user',content=content)],questions=questions(segment))
    client=DecisionsClient(max_calls=1,timeout=150,retries=0,validator=previous.validate)
    reserve();print(json.dumps(dict(stage='request',case=meta['id'],segment=segment+1,layout=layout,repeat=repeat)),flush=True)
    try:response,transport=client.score(payload)
    except Exception as e:atomic_json(folder/'error.json',dict(type=type(e).__name__,message=str(e)));raise
    answers=previous.parsed(response)
    result=dict(identity=identity,response=response,transport=transport,answers=answers,created_unix=time.time())
    atomic_json(receipt,result)
    print(json.dumps(dict(stage='received',case=meta['id'],segment=segment+1,layout=layout,repeat=repeat,answers={k:v['raw'] for k,v in answers.items()})),flush=True)
    return result

def scalar(labels):
    values={'wrong':0.,'partial':.5,'correct':1.}
    v=sum(w*values[l] for w,l in zip([.2,.4,.4],labels))
    if labels[0]=='wrong':v=min(v,.25)
    if labels[2]=='wrong':v=min(v,.5)
    return v

def summarize():
    rs=[json.loads(p.read_text()) for p in (ROOT/'requests').glob('*/receipt.json')]
    lookup={(r['identity']['case']['id'],r['identity']['segment'],r['identity']['layout'],r['identity']['repeat']):r for r in rs}
    dims=list(previous.DIMENSIONS);metrics={k:[] for k in ['positive_controls','late_freeze','late_reverse','early_controls','real_order_agreement','repeat_agreement','stable_real_cells','strict_trajectory_coverage']}
    cases=json.loads((ROOT/'cases.json').read_text());bounds=[]
    real=lambda c:c['kind'] not in ['gt','freeze','reverse_late','matched_pause']
    for r in rs:
        i=r['identity'];kind=i['case']['kind'];s=i['segment'];a=r['answers']
        if kind in ['gt','matched_pause']:metrics['positive_controls'].extend(a[f's{s+1}_{d}']['accepted']=='correct' for d in dims)
        if kind=='freeze' and s>=2:metrics['late_freeze'].append(a[f's{s+1}_dynamics']['accepted']=='wrong')
        if kind=='reverse_late' and s>=2:metrics['late_reverse'].append(a[f's{s+1}_dynamics']['accepted'] in ['wrong','partial'])
        if kind in ['freeze','reverse_late'] and s<2:metrics['early_controls'].extend(a[f's{s+1}_{d}']['accepted']=='correct' for d in dims)
        if i['layout']=='reference_left' and i['repeat']==0:
            for layout,repeat,name in [('prediction_left',0,'real_order_agreement'),('reference_left',1,'repeat_agreement')]:
                b=lookup.get((i['case']['id'],s,layout,repeat))
                if b and (name!='real_order_agreement' or real(i['case'])):metrics[name].extend(a[f's{s+1}_{d}']['raw']==b['answers'][f's{s+1}_{d}']['raw'] for d in dims)
    for c in cases:
        if not real(c):continue
        segments=[]
        for s in range(4):
            a=lookup.get((c['id'],s,'reference_left',0));b=lookup.get((c['id'],s,'prediction_left',0));repeat=lookup.get((c['id'],s,'reference_left',1))
            if not a or not b:continue
            options=[];labels={}
            for d in dims:
                accepted=[x['answers'][f's{s+1}_{d}']['accepted'] for x in [a,b]+([repeat] if repeat else [])]
                stable=len(set(accepted))==1 and accepted[0]!='uncertain';metrics['stable_real_cells'].append(stable)
                labels[d]=accepted[0] if stable else 'uncertain';options.append([accepted[0]] if stable else ['wrong','partial','correct'])
            vs=[scalar(v) for v in itertools.product(*options)];segments.append(dict(segment=s+1,labels=labels,lower=min(vs),upper=max(vs)))
        if len(segments)==4:
            bounds.append(dict(id=c['id'],cohort=c['cohort'],scene=c['scene'],kind=c['kind'],group=c['group'],lower=sum(s['lower'] for s in segments)/4,upper=sum(s['upper'] for s in segments)/4,segments=segments))
            for layout in ['reference_left','prediction_left']:
                aa={k:v for s in range(4) for k,v in lookup[c['id'],s,layout,0]['answers'].items()}
                metrics['strict_trajectory_coverage'].append(previous.provisional_reward(aa) is not None)
    pairwise=[]
    for scene in [4,5]:
        bs=[b for b in bounds if b['scene']==scene]
        for a,b in itertools.combinations(bs,2):pairwise.append(dict(scene=scene,A=a['id'],B=b['id'],winner=a['id'] if a['lower']>b['upper']+1e-9 else b['id'] if b['lower']>a['upper']+1e-9 else None))
    def fraction(v):return dict(numerator=sum(v),denominator=len(v),fraction=sum(v)/len(v) if v else None)
    metrics={k:fraction(v) for k,v in metrics.items()}
    thresholds=dict(positive_controls=.9,late_freeze=.75,late_reverse=.75,early_controls=.9,real_order_agreement=.85,repeat_agreement=.85,stable_real_cells=.75)
    gates={k:metrics[k]['fraction'] is not None and metrics[k]['fraction']>=v for k,v in thresholds.items()}
    gates['informative_pair_in_each_validation_window']=all(any(p['winner'] and p['scene']==s for p in pairwise) for s in [4,5])
    summary=dict(completed_requests=len(rs),planned_requests=196,metrics=metrics,gates=gates,calibration_passed=len(rs)==196 and all(gates.values()),bounds=bounds,validation_pairs=pairwise,optimizer_updates=0)
    atomic_json(ROOT/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['bounds','validation_pairs']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--summary',action='store_true');args=p.parse_args()
    if args.summary:summarize();return
    cases=json.loads((ROOT/'cases.json').read_text())
    protocol=dict(version=2,prompt=PROMPT,rubric_source=str(OLD/'protocol.json'),questions=previous.questions(),model=MODEL,evidence='Shared initial pair, contact sheet of all32 paired timestamps, full-resolution 8 paired TARGET frames plus preceding boundary frame; one segment per call.',layouts=['reference_left','prediction_left'],cases=len(cases),planned_requests=196,max_http_attempts=220,repeat='All4 segments of old scene1/2/3 RL/ReCAP in reference_left once',confidence_thresholds=dict(probability=.6,margin=.2),aggregation='Retain only confidently agreed dimensions across layouts/repeats; enumerate all classes for uncertain dimensions to give deterministic bounds. Accept strict preference only for nonoverlapping bounds. No uncertain-to-zero conversion.',gates=dict(positive_controls=.9,late_freeze=.75,late_reverse=.75,early_controls=.9,real_order_agreement=.85,repeat_agreement=.85,stable_real_cells=.75,informative_pair_in_each_validation_window=True),validation='First2 existing validation entries, disjoint from RL training episodes; now used for judge development, not final test.',optimizer_updates=0)
    atomic_json(ROOT/'protocol.json',protocol);api_environment()
    jobs=[(c,s,l,0) for c in cases for s in range(4) for l in ['reference_left','prediction_left']]
    jobs += [(c,s,'reference_left',1) for c in cases if c['kind']=='rl_recap' for s in range(4)]
    assert len(jobs)==196
    if args.smoke:jobs=[j for j in jobs if (j[0]['id']=='scene1_rl_recap' and j[1]>=2 and j[3]==0) or (j[0]['id']=='val1_freeze' and j[1] in [1,2])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(work,*j) for j in jobs]
        for f in as_completed(fs):f.result()
    summarize()

if __name__=='__main__':main()
