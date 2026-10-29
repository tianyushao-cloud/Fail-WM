#!/usr/bin/env python3
"""CPU-only visual comparison of two reported bad WM forecasts against codec roundtrip."""
import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from action_success_five import encode,decode

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
SRC=ROOT/'previews_wm_codec34k_20261012'
OUT=SRC/'failure_windows';OUT.mkdir(exist_ok=True)
rows={x['id']:x for x in json.loads((ROOT/'robocasa_data/val.json').read_text())['cases']}
for ident,start,end in [('g00006_b04',27,44),('g00014_b06',7,23)]:
    row=rows[ident]
    with np.load(ROOT/'robocasa_data'/row['file']) as z:images=z['images'].copy()
    codec_path=ROOT/'evals_codec34k_fullval_20261011'/f'mixed34k_{ident}.mp4'
    raw=subprocess.run(['ffmpeg','-v','error','-i',str(codec_path),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout
    codec=np.frombuffer(raw,np.uint8).reshape(-1,282,640,3)[:,26:,320:]
    movies=[images[2:],codec,decode(SRC/ident/'success.mp4'),decode(SRC/ident/'natural.mp4')]
    names=['GT','CODEC34K RECONSTRUCTION','WM SUCCESS_ONLY','WM MIXED']
    assert all(x.shape==images[2:].shape for x in movies)
    frames=[]
    for t in range(start,end+1):
        im=Image.new('RGB',(1280,288),'black');d=ImageDraw.Draw(im)
        for col,(movie,name) in enumerate(zip(movies,names)):
            im.paste(Image.fromarray(movie[t-2]),(320*col,32));d.text((320*col+4,3),name,fill='white')
            d.text((320*col+4,17),f'{ident} t={t}',fill='yellow')
        frames.append(np.asarray(im))
    encode(np.stack(frames),OUT/f'{ident}_真实_codec_仅成功WM_混合WM.mp4')
    chosen=list(range(0,len(frames),2))
    sheet=Image.new('RGB',(1280,len(chosen)*288))
    for j,k in enumerate(chosen):sheet.paste(Image.fromarray(frames[k]),(0,j*288))
    sheet.save(OUT/f'{ident}_window.jpg',quality=95)
    print(ident,'window',start,end,flush=True)
