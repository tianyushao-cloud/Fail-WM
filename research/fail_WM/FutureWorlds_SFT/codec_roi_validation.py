"""Distributed validation on fixed real event windows, shared by old/new codecs."""
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from spatial_outcome_metrics import point, distance, references, endpoint

EVENTS={'contact','grasp','lift','drop','release','grasp_attempt_failed'}
SOURCE=Path('/media/tyshao/fail_WM/RoboCasa/data/batch1000/manifest.jsonl')

@lru_cache(maxsize=1)
def events_by_id():
    return {r['episode_id']:r['events'] for r in map(json.loads,SOURCE.read_text().splitlines())}

def positions(row, length):
    ev=[math.ceil(e['state_index']/4) for e in events_by_id()[row['id']] if e['event'] in EVENTS and 'state_index' in e]
    return sorted({length-1}|{t for e in ev for t in range(e-2,e+5) if 2<=t<length})

def roi_mask(truth):
    x=truth.astype(np.float32);red=(x[...,0]>75)&(x[...,0]>1.5*x[...,1])&(x[...,0]>1.4*x[...,2]);y,xx=np.where(red)
    mask=np.zeros(truth.shape[:2],bool)
    if len(xx)>=8:mask[max(0,y.min()-8):min(256,y.max()+9),max(0,xx.min()-8):min(320,xx.max()+9)]=True
    return mask

@torch.inference_mode()
def evaluate(model,data_root,rows,device):
    model.eval();rank=dist.get_rank() if dist.is_initialized() else 0;world=dist.get_world_size() if dist.is_initialized() else 1
    local=[]
    for row in rows[rank::world]:
        with np.load(Path(data_root)/row['file'],allow_pickle=False) as z:real=z['images'].copy()
        times=positions(row,len(real));mae=[];mse=[];roi=[];errors=[];visible=missed=0;last=None
        for begin in range(0,len(times),8):
            ts=times[begin:begin+8]
            batch=torch.from_numpy(real[[0]+ts]).permute(0,3,1,2).float().to(device)/255
            ctx,dyn=model.tokenize(batch[None].contiguous());pred=model.detokenize(ctx,dyn)[0,1:].float().clamp(0,1)
            rgb=np.rint(pred.permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
            for t,rec in zip(ts,rgb):
                truth=real[t];delta=rec.astype(np.float32)-truth.astype(np.float32);mask=roi_mask(truth)
                mae.append(float(np.abs(delta).mean()));mse.append(float(np.square(delta/255).mean()))
                if mask.any():roi.append(float(np.abs(delta)[mask].mean()))
                a,b=point(truth)['xy'],point(rec)['xy']
                visible+=a is not None;missed+=a is not None and b is None
                if a is not None and b is not None:errors.append(distance(a,b))
                if t==len(real)-1:last=rec
        refs=references(real);last_point=point(last);ep=endpoint(last_point['xy'],refs);ds=ep['distances']
        eligible=refs['real_to_goal'] is not None and refs['real_to_goal']>=8 and refs['start_to_goal'] is not None and refs['start_to_goal']>=8
        goalward=bool(eligible and ds['goal'] is not None and ds['goal']+2<min(ds['real'],ds['start']))
        local.append({'id':row['id'],'success':row['success'],'failure_type':row.get('failure_type'),
                      'frames':len(times),'roi_mae_255':float(np.mean(roi)) if roi else None,
                      'mae_255':float(np.mean(mae)),'mse':float(np.mean(mse)),
                      'center_error':float(np.mean(errors)) if errors else None,'visible_frames':visible,'missed_frames':missed,
                      'last_class':ep['class'],'last_missing':last_point['xy'] is None,'goalward_eligible':bool(eligible),'goalward':goalward,
                      'last_distances':ds})
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,local);records=[r for part in gathered for r in part]
    else:records=local
    result={}
    for name,ss in [('success',[r for r in records if r['success']]),('failure',[r for r in records if not r['success']])]:
        roi=[r['roi_mae_255'] for r in ss if r['roi_mae_255'] is not None];centers=[r['center_error'] for r in ss if r['center_error'] is not None]
        vis=sum(r['visible_frames'] for r in ss);miss=sum(r['missed_frames'] for r in ss)
        result[name]={'episodes':len(ss),'frames':sum(r['frames'] for r in ss),'roi_mae_255':float(np.mean(roi)) if roi else None,
                      'mae_255':float(np.mean([r['mae_255'] for r in ss])) if ss else None,
                      'psnr':float(-10*np.log10(max(1e-12,np.mean([r['mse'] for r in ss])))) if ss else None,
                      'center_error':float(np.mean(centers)) if centers else None,
                      'visible_frames':vis,'missed_frames':miss,'miss_rate':miss/vis if vis else None,
                      'last_missing':sum(r['last_missing'] for r in ss),'goalward_eligible':sum(r['goalward_eligible'] for r in ss),
                      'goalward_count':sum(r['goalward'] for r in ss)}
    result['episodes']=records
    result['protocol']='event-2..event+4 plus terminal, red bbox padded8, episode-macro ROI, frame-micro miss rate; success/failure separate'
    return result
