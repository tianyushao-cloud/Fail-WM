import json, math, hashlib, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT=ROOT/'evals_codec34_vs38_fullval_20261024'
r=json.loads((OUT/'report.json').read_text())
assert r['state']=='COMPLETE'
a=r['codecs']['mixed34k'];b=r['codecs']['mixed38k']
aa={x['id']:x for x in a['episodes']};bb={x['id']:x for x in b['episodes']}
assert len(aa)==len(bb)==100 and set(aa)==set(bb)
old=json.loads((ROOT/'evals_codec34k_fullval_20261011/report.json').read_text())['codecs']['mixed34k']
oo={x['id']:x for x in old['episodes']}
result=dict(scope='100 existing validation episodes in 10 initial-state groups; no final test set. Reconstruction only, no WM generation.',
    summary={k:v['summary'] for k,v in r['codecs'].items()},
    events={k:v['event_summary'] for k,v in r['codecs'].items()},
    repeat34k=dict(exact_pred_tracks=sum(aa[k]['pred_track']==oo[k]['pred_track'] for k in aa),
                  max_episode_roi_difference=max(abs(aa[k]['roi_mae_255']-oo[k]['roi_mae_255']) for k in aa)),
    paired={},worst_regressions=[],worst_improvements=[])
changes=[]
for eid,x in aa.items():
    y=bb[eid]
    assert x['gt_track']==y['gt_track'] and x['frames']==y['frames']
    changes.append(dict(id=eid,success=x['success'],group=x['initial_state_id'],
        roi_delta=y['roi_mae_255']-x['roi_mae_255'],
        center_delta=(y['center_error_px']-x['center_error_px']) if x['center_error_px'] is not None and y['center_error_px'] is not None else None,
        missed34=len(x['missing_frames']),missed38=len(y['missing_frames']),
        new_missed_frames=sorted(set(y['missing_frames'])-set(x['missing_frames'])),
        fixed_missed_frames=sorted(set(x['missing_frames'])-set(y['missing_frames']))))
for flag,name in [(True,'success'),(False,'failure')]:
    xs=[x for x in changes if x['success']==flag];groups=sorted({x['group'] for x in xs})
    rng=np.random.default_rng(20261024);boot=[]
    for _ in range(4000):
        sample=rng.choice(groups,len(groups),replace=True)
        ds=[x['roi_delta'] for g in sample for x in xs if x['group']==g]
        boot.append(np.mean(ds))
    result['paired'][name]=dict(roi_improved=sum(x['roi_delta']<0 for x in xs),roi_worse=sum(x['roi_delta']>0 for x in xs),
      mean_roi_delta=float(np.mean([x['roi_delta'] for x in xs])),cluster_bootstrap95_roi_delta=np.percentile(boot,[2.5,97.5]).tolist(),
      independent_initial_state_groups=len(groups),new_missing_frames=sum(len(x['new_missed_frames']) for x in xs),
      fixed_missing_frames=sum(len(x['fixed_missed_frames']) for x in xs))
result['worst_regressions']=sorted(changes,key=lambda x:x['roi_delta'],reverse=True)[:10]
result['worst_improvements']=sorted(changes,key=lambda x:x['roi_delta'])[:10]
result['paired_episodes']=changes
result['threshold_sensitivity']={}
for label,v in r['codecs'].items():
    xs=[x['event']['pred_rise_px'] for x in v['episodes'] if x.get('event',{}).get('category')=='unlifted_failure']
    result['threshold_sensitivity'][label]={str(t):sum(x is not None and x>t for x in xs) for t in [18,r['rise_threshold_px'],25,30,40]}

# Review preidentified residuals, both extremes of ROI change and every newly
# introduced missing-object episode. Selection rules are retained in the result.
selected=['g00006_b00','g00006_b03','g00006_b04','g00014_b06','g00059_b03','g00098_b03','g00071_b05','g00022_b06']
selected += [x['id'] for x in result['worst_regressions'][:3]]+[x['id'] for x in result['worst_improvements'][:2]]
selected=list(dict.fromkeys(selected))
manifest=ROOT/'robocasa_data/val.json'
rows={x['id']:x for x in json.loads(manifest.read_text())['cases']}
gallery=OUT/'comparisons';gallery.mkdir(exist_ok=True)
review=[]
for eid in selected:
    x=aa[eid];y=bb[eid]
    with np.load(manifest.parent/rows[eid]['file'],allow_pickle=False) as z:gt=z['images']
    with np.load(OUT/'frames/mixed34k'/(eid+'.npz')) as z:im34=z['reconstruction']
    with np.load(OUT/'frames/mixed38k'/(eid+'.npz')) as z:im38=z['reconstruction']
    assert len(im34)==len(im38)==len(gt)-2
    # Include frames with the worst baseline/new center errors, missed detections,
    # the largest RGB change between codecs, and the final frame.
    times=[]
    for item in [x,y]:
        errs=item['center_errors_px'];order=sorted(range(len(errs)),key=lambda i:errs[i] if errs[i] is not None else -1,reverse=True)
        times.extend([order[0]+2]);times.extend(item['missing_frames'][:1])
    if eid=='g00059_b03':times.extend([54,57,61])
    if eid=='g00098_b03':times.extend([66,67,68])
    dif=np.abs(im38.astype(float)-im34.astype(float)).mean((1,2,3))
    times.extend([int(np.argmax(dif))+2,len(gt)-1]);times=sorted(set(times))[:8]
    # Fixed scene detail box covers object, gripper and goal for both codecs and GT.
    allgt=[p for p in x['gt_track'] if p is not None]
    px=[p[0] for p in allgt];py=[p[1] for p in allgt]
    box=(max(0,int(min(px))-45),max(0,int(min(py))-55),min(320,int(max(px))+45),min(256,int(max(py))+35))
    for detail in [False,True]:
        width=320;height=256
        sheet=Image.new('RGB',(width*len(times),3*(height+28)),(20,20,20));draw=ImageDraw.Draw(sheet)
        for j,t in enumerate(times):
            for i,(lab,arr) in enumerate([('GT',gt[t]),('34k',im34[t-2]),('38k',im38[t-2])]):
                im=Image.fromarray(arr)
                if detail:im=im.crop(box).resize((width,height),Image.Resampling.NEAREST)
                sheet.paste(im,(j*width,i*(height+28)+28))
                draw.text((j*width+5,i*(height+28)+6),f'{eid} {lab} t={t}'+(' DETAIL' if detail else ''),fill='white')
        sheet.save(gallery/(eid+('_detail' if detail else '')+'.png'))
    review.append(dict(id=eid,times=times,detail_box=box,success=x['success'],roi_delta=next(c['roi_delta'] for c in changes if c['id']==eid)))
result['visual_review_selection']=review
result['full_frame_reconstruction_count_per_codec']=sum(x['frames'] for x in a['episodes'])
result['interpretation_limits']=['Red-color detector is a proxy; missing detections are not necessarily physical disappearance.',
 'Fidelity of the event must be confirmed visually; reduced ROI error alone is not sufficient.',
 'The validation set has been used for prior checkpoint selection; bootstrap is exploratory, not independent final-test evidence.']
(OUT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['paired_episodes','worst_regressions','worst_improvements','visual_review_selection']},indent=2))
