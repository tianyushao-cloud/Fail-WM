#!/usr/bin/env python3
"""Reuse the frozen Opus codec audit with a bounded CUDA allocator."""
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
from codec_fake_lift_audit import roundtrip, CompressiveVQModelFSQ, CodecWorker, read_case, point, lift_metrics, release_metrics

OUT = ROOT / 'evals_codec_current_20261011'
OUT.mkdir(exist_ok=True)
if (OUT / 'report.json').exists():
    raise RuntimeError('This acceptance report is frozen. Use a new output directory for another audit.')
selpath = ROOT / 'evals_gt_handoff_20261010/selection.json'
sel = json.loads(selpath.read_text())
rows = {r['id']:r for r in json.loads((ROOT / 'robocasa_data/val.json').read_text())['cases']}
report = dict(rise_threshold_px=sel['rise_threshold_px'], selection=str(selpath),
              selection_sha256=hashlib.sha256(selpath.read_bytes()).hexdigest(),
              precision=dict(dtype='float32',matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False),
              encode='SFT cache chunks of 8; original full episode', decode='CodecWorker pairs of 2; frames 2 onward', codecs={})
specs = [(f'mixed_{s}', ROOT / f'runs/paper_adapt_mixed_codec_20261009/checkpoints/step_{s:06d}') for s in (34000,32000,30000)]
specs += [('old_success_44000', ROOT / 'assets/futureworlds_bundle_robocasa_codec_10k/robocasa/codec')]
for label,path in specs:
    codec = CompressiveVQModelFSQ.from_pretrained(path, local_files_only=True).to('cuda').eval()
    worker = CodecWorker(codec,None)
    per=[]; started=time.monotonic(); preview_categories=set()
    for i,c in enumerate(sel['cases']):
        imgs,_,_ = read_case(ROOT / 'robocasa_data/val.json', rows[c['id']], 52)
        recimgs = roundtrip(codec,worker,imgs,'cuda')
        gt=[point(f)['xy'] for f in imgs]; rec=[point(f)['xy'] for f in recimgs]
        m = (release_metrics(rec,gt,2,c['end']) if c['category']=='release_failure'
             else lift_metrics(rec,gt,2,c['anchor'],c['end'],sel['rise_threshold_px']))
        per.append(dict(id=c['id'],category=c['category'],failure_type=c['failure_type'],factors=c['factors'],
                        track=rec,gt_track=gt,**m))
        preview = c['category'] not in preview_categories or c['id'] in ('g00014_b06','g00059_b03')
        if preview or (c['category']=='unlifted_failure' and m['pred_lifted']):
            indices=sorted(set(int(x) for x in np.linspace(max(2,c['anchor']-1),c['end'],7)))
            canvas=Image.new('RGB',(320*len(indices),2*(256+25)),(20,20,20));d=ImageDraw.Draw(canvas)
            for col,t in enumerate(indices):
                for row,arr,name in ((0,imgs[t],'GT'),(1,recimgs[t-2],label)):
                    x,y=320*col,row*281
                    canvas.paste(Image.fromarray(arr),(x,y+25));d.text((x+4,y+5),f'{name} t={t}',fill='white')
            canvas.save(OUT/f'{label}_{c["id"]}.jpg',quality=92)
        if preview:
            frames=[]
            for t in range(2,len(imgs)):
                frame=Image.new('RGB',(640,282),(20,20,20)); d=ImageDraw.Draw(frame)
                frame.paste(Image.fromarray(imgs[t]),(0,25));frame.paste(Image.fromarray(recimgs[t-2]),(320,25))
                d.text((6,6),f'GT {c["id"]} t={t}',fill='white');d.text((326,6),label,fill='white')
                frames.append(np.asarray(frame))
            subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                            '-s','640x282','-r','10','-i','pipe:0','-an','-c:v','libx264',
                            '-threads','2','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart',
                            str(OUT/f'{label}_{c["id"]}.mp4')],input=np.stack(frames).tobytes(),check=True)
            preview_categories.add(c['category'])
        if (i+1)%5==0 or i+1==len(sel['cases']):
            print(label,f'{i+1}/{len(sel["cases"])}',round(time.monotonic()-started,1),'seconds',flush=True)
    summary={}
    for cat in ('unlifted_failure','success_control','release_failure'):
        xs=[x for x in per if x['category']==cat]
        if cat=='release_failure':
            summary[cat]={k:sum(x['class']==k for x in xs) for k in ('fell_like_gt','still_high','vanished','other','gt_missing')}
        else:
            summary[cat]=dict(n=len(xs),lifted=sum(x['pred_lifted'] is True for x in xs),
                              not_lifted=sum(x['pred_lifted'] is False for x in xs),
                              undetected=sum(x['pred_lifted'] is None for x in xs),
                              any_vanished_frame=sum(x['vanished_frames']>0 for x in xs))
    report['codecs'][label]=dict(path=str(path),summary=summary,per_case=per)
    (OUT/f'{label}.json').write_text(json.dumps(report['codecs'][label],indent=2)+'\n')
    (OUT/'report.partial.json').write_text(json.dumps(report,indent=2)+'\n')
    print(label,json.dumps(summary),flush=True)
    del codec,worker;torch.cuda.empty_cache()
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('Peak CUDA allocated MiB:', torch.cuda.max_memory_allocated() / 1024**2, flush=True)
