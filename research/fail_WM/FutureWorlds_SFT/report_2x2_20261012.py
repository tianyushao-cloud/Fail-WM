"""Frozen event cohorts, group-bootstrap differences, and numeric guardrails."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from prepare_2x2_20261012 import ROOT,EXP,EVAL,load,atomic
from futureworlds.checkpoints import file_sha256

def bootstrap(a,b):
    # Each group contributes all of its episodes and repeated decodes together.
    groups=sorted(set(a)|set(b))
    if not groups:return None
    av=np.array([a.get(g,[0,0]) for g in groups],float)
    bv=np.array([b.get(g,[0,0]) for g in groups],float)
    rng=np.random.default_rng(20261012)
    draw=rng.integers(0,len(groups),size=(10000,len(groups)))
    aa=av[draw].sum(1);bb=bv[draw].sum(1)
    valid=(aa[:,1]>0)&(bb[:,1]>0)
    diff=aa[valid,0]/aa[valid,1]-bb[valid,0]/bb[valid,1]
    return dict(groups=len(groups),bootstrap_repeats=10000,valid_repeats=int(valid.sum()),
                difference=float(av[:,0].sum()/av[:,1].sum()-bv[:,0].sum()/bv[:,1].sum()),
                ci95=np.quantile(diff,[.025,.975]).tolist(),unit='initial_state_group',seed=20261012)

def main():
    protocol=load(EXP/'protocol.json');selection=load(EVAL/'selection.json');cases=selection['cases']
    aroot=Path(protocol['A_events'])
    if file_sha256(aroot/'selection.json')!=protocol['selection_sha256']:raise ValueError('A selection differs')
    results={}
    missing=[]
    for model in ('A','B','C','D'):
        folder=aroot/'results/c34_natural' if model=='A' else EVAL/'results'/model
        results[model]={}
        for case in cases:
            path=folder/(case['id']+'.json')
            if not path.exists():missing.append(str(path));continue
            r=load(path);hs={h['label']:h for h in r['handoffs']}
            assert set(hs)=={'start','+2'},path
            for h in hs.values():
                assert len(h['rollouts'])==9
                assert sum(x['decode']=='beam4' for x in h['rollouts'])==1
                assert sum(x['decode']=='sampling' for x in h['rollouts'])==8
            results[model][case['id']]=r
    if missing:
        atomic(EVAL/'comparison_pending.json',dict(state='WAITING_FOR_COMPLETE_BASELINE_OR_EVAL',missing=missing))
        raise RuntimeError(f'{len(missing)} event results missing; rerun same submission command after baseline completion')
    # Require identical event protocol except the explicitly different model list.
    ar=load(aroot/'run_c34_natural.json')
    for model in ('B','C','D'):
        rr=load(EVAL/f'run_{model}.json')
        for key in ('selection_sha256','samples','temperature','beam_width','memory','seed','history_tokens','handoff_labels'):
            assert rr[key]==ar[key],key
    raw={r['episode_id']:r for r in map(json.loads,Path('/media/tyshao/fail_WM/RoboCasa/data/batch1000/manifest.jsonl').read_text().splitlines())}
    cohort=defaultdict(list);disagreements=[]
    for c in cases:
        cid=c['id'];cat=c['category']
        if cat=='success_control':cohort['success_control'].append(cid)
        elif cat=='release_failure':cohort['release_failure'].append(cid)
        else:
            if 'partial_close' in c['factors']:cohort['partial_close'].append(cid)
            flags={m:results[m][cid]['codec_recon']['pred_lifted'] for m in results}
            if len(set(flags.values()))!=1:disagreements.append(dict(id=cid,flags=flags))
            # Fixed intersection keeps the evaluated episodes identical across models.
            if not all(v is False for v in flags.values()):continue
            phase=raw[cid]['phase_ranges']['attempt0/lift']
            with np.load(ROOT/'robocasa_data'/c['file']) as z:
                grip=z['actions'].reshape(-1,13)[phase['first_action']:phase['last_action']+1,6]
            if np.allclose(grip,1):cohort['clean_closed_never_lift'].append(cid)
    stats=[];counts={}
    for name,ids in cohort.items():
        for handoff in ('start','+2'):
            for model in results:
                for decode in ('combined','beam4','sampling'):
                    group=defaultdict(lambda:[0,0]);n=hit=missing=vanished_any=0
                    for cid in ids:
                        r=results[model][cid];h=next(h for h in r['handoffs'] if h['label']==handoff)
                        for x in h['rollouts']:
                            if decode!='combined' and x['decode']!=decode:continue
                            if name=='release_failure':
                                value=x['class']=='vanished';unknown=x['class']=='gt_missing'
                            else:
                                value=x['pred_lifted'] is True;unknown=x['pred_lifted'] is None
                                vanished_any+=x['vanished_frames']>0
                            hit+=value;n+=1;missing+=unknown
                            group[r['group']][0]+=int(value);group[r['group']][1]+=1
                    assert n>0
                    key=(name,handoff,decode,model);counts[key]=dict(group)
                    stats.append(dict(cohort=name,handoff=handoff,decode=decode,model=model,cases=len(ids),
                                      n=n,numerator=hit,rate=hit/n,undetected_or_gt_missing=missing,
                                      rate_among_detected=hit/(n-missing) if n>missing else None,
                                      any_vanished=vanished_any if name!='release_failure' else None))
    comparisons=[]
    for name in cohort:
        for h in ('start','+2'):
            for d in ('combined','beam4','sampling'):
                for m in ('B','C','D'):
                    comparisons.append(dict(cohort=name,handoff=h,decode=d,model=m,
                                            **bootstrap(counts[name,h,d,m],counts[name,h,d,'A'])))
    frame={}
    for model in results:
        s=load(EVAL/'frame_metrics'/model/'summary.json')
        assert s['state']=='COMPLETE' and s['protocol']['horizons']==[10,20]
        assert s['protocol']['manifest_sha256']==protocol['manifest_sha256']['evaluation_manifest']
        assert s['protocol']['bundle_manifest_sha256']==file_sha256(EVAL/'bundles'/model/'manifest.json')
        frame[model]=s['aggregate']
    def rate(model,cohort,h):
        return next(x['rate'] for x in stats if x['model']==model and x['cohort']==cohort and x['handoff']==h and x['decode']=='combined')
    guards={}
    for m in ('B','C','D'):
        g={}
        for h in ('start','+2'):
            g['success_ge_95pct/'+h]=rate(m,'success_control',h)>=.95
            g['partial_close_le_A/'+h]=rate(m,'partial_close',h)<=rate('A','partial_close',h)
            g['release_vanished_le_A/'+h]=rate(m,'release_failure',h)<=rate('A','release_failure',h)
        g['lpips_mean_first20_le_1.05A']=frame[m]['20']['lpips']<=frame['A']['20']['lpips']*1.05
        guards[m]=dict(checks=g,all_pass=all(g.values()))
    result=dict(state='COMPLETE',cohorts=dict(cohort),codec_disagreements=disagreements,
                rates=stats,group_bootstrap=comparisons,frame_metrics=frame,guardrails=guards,
                caveats=['Rates use all outputs as denominator and report undetected separately; a lower lift rate from missing cubes is not demonstrated improvement.',
                         'LPIPS@20 here is the original evaluator mean over the first 20 generated frames, not only frame 20.',
                         'Clean cohort is the intersection of codec_recon.pred_lifted=False across all four models; discrepancies are disclosed.',
                         'Guardrails are checked separately at start and +2. No test set used. Numeric results only; no causal or solved claim.'])
    atomic(EVAL/'comparison.json',result)
    lines=['# codec34k 2×2 数值对照','','原始结果：comparison.json；以初态组bootstrap，10000次。',
           '缺失单列，不把不可见解释成正确预测失败。LPIPS@20是前20个生成帧的平均。','',
           '|组|条件|关键失败虚构抬升|不可判|成功抬升|提前释放消失|','|---|---|---:|---:|---:|---:|']
    for m in results:
        for h in ('start','+2'):
            s=next(x for x in stats if x['model']==m and x['cohort']=='clean_closed_never_lift' and x['handoff']==h and x['decode']=='combined')
            lines.append(f"|{m}|{h}|{s['numerator']}/{s['n']} ({s['rate']:.2%})|{s['undetected_or_gt_missing']}|{rate(m,'success_control',h):.2%}|{rate(m,'release_failure',h):.2%}|")
    lines+=['','逐项护栏与分组置信区间见comparison.json。','',json.dumps(guards,ensure_ascii=False,indent=2)]
    (EVAL/'COMPARISON_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(state='COMPLETE',guardrails=guards),indent=2))

if __name__=='__main__':main()
