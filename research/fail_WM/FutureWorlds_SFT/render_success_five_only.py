#!/usr/bin/env python3
"""Render only; reuse completed videos, no outcome scoring."""
import gc
import json
import subprocess
import time
import shutil
import torch
import numpy as np
from PIL import Image
from action_success_five import ROOT, OUT, MANIFEST, BUNDLES, TEXT, KINDS, interventions, encode, decode, panel, write
from futureworlds.data import read_case
from futureworlds.pipeline import WorldPipeline

LABELS={'success':'SUCCESS_ONLY (443 success)', 'natural':'MIXED (443 success + 357 failure)'}
FILENAMES={'success':'仅成功数据训练_动作对照.mp4','natural':'成功加失败数据训练_动作对照.mp4'}

def main():
    selection=json.loads((OUT/'selection.json').read_text());torch.set_num_threads(2)
    for model in ('success','natural'):
        needed=any(not (OUT/r['id']/model/(k+'.mp4')).exists() for r in selection['cases'] for k in KINDS)
        pipe=WorldPipeline(BUNDLES/(model+'_bundle'),'robocasa','sft','cuda',TEXT) if needed else None
        for row in selection['cases']:
            dest=OUT/row['id']/model;dest.mkdir(parents=True,exist_ok=True)
            images,actions,text=read_case(MANIFEST,row,52);variants,_=interventions(actions);movies={}
            for kind in KINDS:
                path=dest/(kind+'.mp4')
                if not path.exists():
                    torch.manual_seed(0)
                    with torch.inference_mode():pred=pipe.predict(images[:2],variants[kind],text,len(images)-2,4,'full')['prediction']
                    rgb=np.rint(pred*255).clip(0,255).astype(np.uint8);encode(rgb,path);Image.fromarray(rgb[-1]).save(dest/(kind+'_last.png'))
                    print('rendered',model,row['id'],kind,flush=True)
                movies[kind]=decode(path)
            target=OUT/row['id']/('comparison_'+model+'.mp4')
            panel(images[2:],movies,LABELS[model],target)
            shutil.copyfile(target,OUT/row['id']/FILENAMES[model])
            print('comparison_ready',model,row['id'],flush=True)
        del pipe;gc.collect();torch.cuda.empty_cache()
    for row in selection['cases']:
        dest=OUT/row['id'];a=dest/'comparison_success.mp4';b=dest/'comparison_natural.mp4';target=dest/'comparison_both.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(a),'-i',str(b),'-filter_complex','[0:v][1:v]vstack=inputs=2[v]','-map','[v]','-c:v','libx264','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],capture_output=True,check=True)
        top=Image.open(a.with_suffix('.png'));bottom=Image.open(b.with_suffix('.png'));im=Image.new('RGB',(1280,576));im.paste(top,(0,0));im.paste(bottom,(0,288));im.save(target.with_suffix('.png'))
        shutil.copyfile(target,dest/'两模型对照_上仅成功_下成功加失败.mp4')
    write(OUT/'complete.json',{'state':'COMPLETE','mode':'videos_only','episodes':5,'models':2,'predictions':30,'comparisons':5})
    (OUT/'VIDEOS_README.txt').write_text('SUCCESS_ONLY / success / comparison_success.mp4 = 仅成功数据训练（443条成功）。\nMIXED / natural / comparison_natural.mp4 = 成功加失败数据训练（443条成功+357条失败）。\n每个案例另有中文文件名。两模型对照：上排仅成功训练，下排成功加失败训练；每排从左到右：真实视频、原动作预测、随机XYZ扰动、夹爪反转。模型标签说明训练数据，不能理解为这次预测的成功或失败。此次仅供观看，不作成功失败判定。\n')
    print('ALL FIVE VIDEO COMPARISONS COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    finally:subprocess.run(['/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python',str(ROOT/'start_expanded_e1.py')],check=True)
