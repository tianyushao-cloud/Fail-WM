#!/usr/bin/env python3
"""Render the largest full-trajectory errors found by the completed new-codec pass."""
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from futureworlds.pipeline import WorldPipeline
from futureworlds.batch import TensorBatch
from futureworlds.data import read_case

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT');OUT=ROOT/'evals_codec34k_fullval_20261011'
r=json.loads((OUT/'mixed34k.json').read_text());manifest=ROOT/'robocasa_data/val.json'
rows={x['id']:x for x in json.loads(manifest.read_text())['cases']}
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
pipe=WorldPipeline(r['bundle'],'robocasa','sft','cuda',str(ROOT.parent/'FutureWorlds_workspace/assets/t5-base'))
for ident in ('g00071_b05','g00022_b06','g00098_b03'):
    x=next(x for x in r['episodes'] if x['id']==ident)
    es=x['center_errors_px'];t=max(range(len(es)),key=lambda i:es[i] or 0)+2
    with torch.inference_mode():
        imgs,acts,txt=read_case(manifest,rows[ident],52)
        c,d,a,_,cond=pipe.encode_training_window(imgs,acts,txt)
        pix=pipe.worker.detokenize(TensorBatch({'tokens':d[2:][None],'ctx_tokens':c}),None).batch['pixels'][0,1:]
        rec=np.rint(pix.permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
    times=sorted({max(2,t-1),t,min(len(imgs)-1,t+1),*x['missing_frames'][:1],len(imgs)-1})
    im=Image.new('RGB',(320*len(times),564),(20,20,20));draw=ImageDraw.Draw(im)
    for col,frame in enumerate(times):
        for row,arr,title in ((0,imgs[frame],'GT'),(1,rec[frame-2],'mixed34k')):
            im.paste(Image.fromarray(arr),(col*320,row*282+26));draw.text((col*320+5,row*282+5),f'{title} {ident} t={frame}',fill='white')
    im.save(OUT/f'mixed34k_{ident}_worst.png')
    cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x282','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'mixed34k_{ident}_full.mp4')]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for frame,arr in enumerate(rec,2):
        f=Image.new('RGB',(640,282),(20,20,20));draw=ImageDraw.Draw(f)
        f.paste(Image.fromarray(imgs[frame]),(0,26));f.paste(Image.fromarray(arr),(320,26))
        draw.text((5,5),f'GT {ident} t={frame}',fill='white');draw.text((325,5),'mixed34k',fill='white');p.stdin.write(np.asarray(f).tobytes())
    p.stdin.close();assert p.wait()==0
    print(ident,'worst frame',t,flush=True)
