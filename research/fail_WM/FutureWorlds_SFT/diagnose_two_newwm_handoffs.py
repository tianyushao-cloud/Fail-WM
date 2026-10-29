#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from futureworlds.pipeline import WorldPipeline
from futureworlds.data import read_case
from futureworlds.memory import make_prefix
from futureworlds.decoding import beam_decode
from futureworlds.batch import TensorBatch
from action_success_five import encode,decode,write
from spatial_outcome_metrics import point

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT');BASE=ROOT/'previews_wm_codec34k_20261012'
OUT=BASE/'failure_windows/gt_handoff';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory)
bundle=BASE/'natural_bundle';pipe=WorldPipeline(bundle,'robocasa','sft','cuda','/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base')
manifest=ROOT/'robocasa_data/val.json';rows={r['id']:r for r in json.loads(manifest.read_text())['cases']};results=[]
for ident,k,end in [('g00006_b04',34,45),('g00014_b06',17,27)]:
    imgs,actions,text=read_case(manifest,rows[ident],52)
    with torch.inference_mode():
        ctx,dyn,acts,_,_=pipe.encode_training_window(imgs,actions,text);pipe.condition(text)
        history={1:dyn[1:2],**{t:dyn[t:t+1] for t in range(max(2,k-6),k)}};responses=[]
        for target in range(k,end+1):
            prefix,_=make_prefix(ctx.reshape(1,-1)+4375,history,acts,target)
            candidates,_,_=beam_decode(pipe.model,prefix,width=4)
            history[target]=candidates[:1];responses.append(candidates[:1]);history={t:v for t,v in history.items() if t==1 or t>=target-5}
        codes=torch.stack(responses,1);start=k-k%2
        # Keep the original even/odd decode pair alignment.
        padded=torch.cat([dyn[start:k][None],codes],1)
        pix=pipe.worker.detokenize(TensorBatch({'tokens':padded,'ctx_tokens':ctx}),None).batch['pixels'][0,1:]
        movie=np.rint(pix.permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)[k-start:]
    openloop=decode(BASE/ident/'natural.mp4');frames=[]
    for i,t in enumerate(range(k,end+1)):
        frame=Image.new('RGB',(960,288),'black');d=ImageDraw.Draw(frame)
        for col,(pic,title) in enumerate([(imgs[t],'GT'),(openloop[t-2],'MIXED OPEN LOOP'),(movie[i],f'MIXED GT HISTORY UNTIL t{k-1}')]):
            frame.paste(Image.fromarray(pic),(col*320,32));d.text((col*320+4,3),title,fill='white');d.text((col*320+4,17),f'{ident} target {t}',fill='yellow')
        frames.append(np.asarray(frame))
    encode(np.stack(frames),OUT/f'{ident}_late_history.mp4')
    sheet=Image.new('RGB',(960,288*len(frames)))
    for i,frame in enumerate(frames):sheet.paste(Image.fromarray(frame),(0,i*288))
    sheet.save(OUT/f'{ident}_all_frames.jpg',quality=95)
    result=dict(id=ident,k=k,end=end,history='GT anchor1+recent6 before k; autoregressive thereafter',beam=4,
                gt=[point(f)['xy'] for f in imgs[k:end+1]],openloop=[point(f)['xy'] for f in openloop[k-2:end-1]],
                late=[point(f)['xy'] for f in movie],codec_bundle=str(bundle),
                caveat='Two-case diagnostic; uses GT future history unavailable in normal rollout. Pairwise decode aligned to absolute even frames. No causal attribution from rate.')
    results.append(result);write(OUT/'results.json',results);print(ident,'complete',flush=True)
