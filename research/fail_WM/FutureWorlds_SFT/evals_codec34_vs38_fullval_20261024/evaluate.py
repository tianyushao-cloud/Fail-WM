#!/usr/bin/env python3
"""Independent deployed-pipeline, full-validation RGB roundtrip comparison."""
import gc
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from futureworlds.pipeline import WorldPipeline
from futureworlds.batch import TensorBatch
from futureworlds.data import read_case
from futureworlds.checkpoints import file_sha256
from codec_roi_validation import roi_mask
from spatial_outcome_metrics import point, distance
from gt_handoff import lift_metrics, release_metrics

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT=ROOT/'evals_codec34_vs38_fullval_20261024'
OUT.mkdir(exist_ok=True)
if (OUT/'report.json').exists():
    raise RuntimeError('Completed report is immutable; use a new directory')
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
manifest=ROOT/'robocasa_data/val.json'
rows=json.loads(manifest.read_text())['cases']
selection_path=ROOT/'evals_gt_handoff_20261010/selection.json'
selection=json.loads(selection_path.read_text())
events={c['id']:c for c in selection['cases']}
expected=json.loads((ROOT/'experiments_mixed_codec_wm_20261011/codec_selection.json').read_text())
prior=json.loads((ROOT/'evals_codec_current_20261011/report.json').read_text())['codecs']['mixed_34000']
preview_ids={'g00006_b00','g00006_b03','g00006_b04','g00014_b06','g00014_b07','g00059_b03','g00098_b03','g00071_b05','g00022_b06'}
report=dict(protocol='Actual WorldPipeline.encode_training_window + CodecWorker.detokenize pairs2; all frames 2..T-1; FP32/TF32off; no WM generation',
            manifest_sha256=file_sha256(manifest),selection_sha256=file_sha256(selection_path),
            rise_threshold_px=selection['rise_threshold_px'],codecs={})

def save(path,data):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def video(label,ident,real,rec):
    cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x282','-r','5','-i','pipe:0','-an',
         '-c:v','libx264','-threads','2','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'{label}_{ident}.mp4')]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for t,img in enumerate(rec,2):
        frame=Image.new('RGB',(640,282),(15,15,15));d=ImageDraw.Draw(frame)
        frame.paste(Image.fromarray(real[t]),(0,26));frame.paste(Image.fromarray(img),(320,26))
        d.text((5,5),f'GT {ident} t={t}',fill='white');d.text((325,5),label,fill='white')
        p.stdin.write(np.asarray(frame).tobytes())
    p.stdin.close()
    if p.wait():raise RuntimeError('ffmpeg failed')

def strip(label,ident,real,rec,times,name):
    times=sorted(set(t for t in times if 2<=t<len(real)))[:8]
    im=Image.new('RGB',(320*len(times),564),(15,15,15));d=ImageDraw.Draw(im)
    for col,t in enumerate(times):
        for row,arr,title in ((0,real[t],'GT'),(1,rec[t-2],label)):
            im.paste(Image.fromarray(arr),(col*320,row*282+26));d.text((col*320+5,row*282+5),f'{title} {ident} t={t}',fill='white')
    im.save(OUT/f'{label}_{ident}_{name}.png')

for label,bundle in [('mixed34k',Path(expected['bundle'])),('mixed38k',ROOT/'assets/futureworlds_bundle_mixed_codec_20261009')]:
    ck=bundle/'robocasa/codec/diffusion_pytorch_model.safetensors'
    digest=file_sha256(ck)
    if label=='mixed34k':
        assert digest==expected['codec_sha256']==file_sha256(Path(expected['selected']['checkpoint'])/'diffusion_pytorch_model.safetensors')
        for variant,prefix in [('success','success_only'),('natural','natural')]:
            cfg=json.loads((ROOT/f'runs/{prefix}_sft_mixed_codec34000_10k_20261011/run.json').read_text())['config']
            assert cfg['bundle']==str(bundle)
    if label=='mixed38k':
        assert digest=='35b1522d54d95b7e1dddcb3dbe15c3a99e2874bb1468aa175711ed22956bd14a'
    pipe=WorldPipeline(str(bundle),'robocasa','sft','cuda',str(ROOT.parent/'FutureWorlds_workspace/assets/t5-base'))
    # WM model is loaded for exact pipeline construction, but never run here.
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    print('START',label,flush=True)
    records=[];start=time.monotonic()
    for i,row in enumerate(rows):
        images,actions,text=read_case(manifest,row,pipe.action_dim)
        with torch.inference_mode():
            ctx,dyn,acts,_,cond=pipe.encode_training_window(images,actions,text)
            pixels=pipe.worker.detokenize(TensorBatch({'tokens':dyn[2:][None],'ctx_tokens':ctx}),None).batch['pixels'][0,1:]
            rec=np.rint(pixels.permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
        assert rec.shape==images[2:].shape and dyn.shape==(len(images),80)
        frame_dir=OUT/'frames'/label
        frame_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(frame_dir/(row['id']+'.npz'),reconstruction=rec)
        gt=[point(f)['xy'] for f in images];pred=[point(f)['xy'] for f in rec]
        roi=[];mae=[];mse=[];errs=[];missing=[];hallucinated=[]
        for t,r in enumerate(rec,2):
            delta=r.astype(np.float32)-images[t].astype(np.float32);mask=roi_mask(images[t])
            mae.append(float(np.abs(delta).mean()));mse.append(float(np.square(delta/255).mean()))
            if mask.any():roi.append(float(np.abs(delta)[mask].mean()))
            e=distance(gt[t],pred[t-2]);errs.append(e)
            if gt[t] is not None and pred[t-2] is None:missing.append(t)
            if gt[t] is None and pred[t-2] is not None:hallucinated.append(t)
        finite=[e for e in errs if e is not None]
        item=dict(id=row['id'],initial_state_id=row['initial_state_id'],success=row['success'],failure_type=row.get('failure_type'),frames=len(rec),
                  mae_255=float(np.mean(mae)),mse=float(np.mean(mse)),roi_mae_255=float(np.mean(roi)) if roi else None,
                  gt_visible_frames=sum(p is not None for p in gt[2:]),missing_frames=missing,
                  detection_when_gt_missing=hallucinated,center_error_px=float(np.mean(finite)) if finite else None,
                  max_center_error_px=max(finite) if finite else None,center_errors_px=errs,
                  gt_track=gt,pred_track=pred,last_gt_visible=gt[-1] is not None,last_pred_visible=pred[-1] is not None)
        c=events.get(row['id']);event=None
        if c:
            event=(release_metrics(pred,gt,2,c['end']) if c['category']=='release_failure' else
                   lift_metrics(pred,gt,2,c['anchor'],c['end'],selection['rise_threshold_px']))
            item['event']=dict(category=c['category'],**event)
        if label=='mixed34k' and c:
            prev=next(x for x in prior['per_case'] if x['id']==row['id'])
            item['previous_audit_track_exact']=prev['track']==pred
        if row['id'] in preview_ids:video(label,row['id'],images,rec)
        worst=sorted(range(2,len(images)),key=lambda t:errs[t-2] if errs[t-2] is not None else -1,reverse=True)[:2]
        times=worst+missing[:2]+[len(images)-1]
        if c:
            times += [max(2,c['anchor']-1),c['anchor'],c['end']]
        if row['id'] in preview_ids or (c and ((c['category']=='unlifted_failure' and event['pred_lifted']) or (c['category']=='release_failure' and event['class']!='fell_like_gt'))):
            strip(label,row['id'],images,rec,times,'audit')
        records.append(item)
        save(OUT/f'{label}_progress.json',dict(completed=len(records),total=len(rows),last_id=row['id']))
        if (i+1)%5==0:print(label,i+1,'/',len(rows),round(time.monotonic()-start,1),'seconds',flush=True)
    summary={}
    for group,flag in [('success',True),('failure',False)]:
        xs=[x for x in records if x['success']==flag];vis=sum(x['gt_visible_frames'] for x in xs);miss=sum(len(x['missing_frames']) for x in xs)
        es=[e for x in xs for e in x['center_errors_px'] if e is not None]
        summary[group]=dict(episodes=len(xs),frames=sum(x['frames'] for x in xs),
                            roi_mae_255=float(np.mean([x['roi_mae_255'] for x in xs if x['roi_mae_255'] is not None])),
                            psnr_from_episode_mean_mse=-10*math.log10(np.mean([x['mse'] for x in xs])),
                            center_error_px=float(np.mean(es)),center_error_p95_px=float(np.percentile(es,95)),
                            gt_visible_frames=vis,missed_frames=miss,miss_rate=miss/vis,
                            gt_missing_but_pred_detected=sum(len(x['detection_when_gt_missing']) for x in xs),
                            endpoint_miss_when_gt_visible=sum(x['last_gt_visible'] and not x['last_pred_visible'] for x in xs))
    event_summary={}
    for cat in ('unlifted_failure','success_control','release_failure'):
        xs=[x['event'] for x in records if x.get('event',{}).get('category')==cat]
        if cat=='release_failure':
            event_summary[cat]={k:sum(x['class']==k for x in xs) for k in ('fell_like_gt','still_high','vanished','other','gt_missing')}
        else:
            event_summary[cat]=dict(n=len(xs),lifted=sum(x['pred_lifted'] is True for x in xs),
                                    undetected=sum(x['pred_lifted'] is None for x in xs),
                                    missing_frames=sum(x['vanished_frames'] for x in xs),
                                    gt_visible_frames=sum(x['gt_visible_frames'] for x in xs),
                                    episodes_with_missing=sum(x['vanished_frames']>0 for x in xs))
    result=dict(bundle=str(bundle),codec_sha256=digest,summary=summary,event_summary=event_summary,episodes=records)
    if label=='mixed34k':result['previous_audit_exact_tracks']=sum(x.get('previous_audit_track_exact',False) for x in records)
    report['codecs'][label]=result;save(OUT/f'{label}.json',result);save(OUT/'report.partial.json',report)
    print(label,json.dumps(summary),json.dumps(event_summary),flush=True)
    del pipe,ctx,dyn,acts,pixels,cond;gc.collect();torch.cuda.empty_cache()
report['state']='COMPLETE';save(OUT/'report.json',report)
print('COMPLETE',flush=True)
