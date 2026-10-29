#!/usr/bin/env python3
"""Five GT-success cases, unchanged original action interventions, two old WMs."""
import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from futureworlds.data import read_case, read_manifest
from futureworlds.pipeline import WorldPipeline, encode_actions
from action_intervention_new_codec import interventions, START, END, SEED
from spatial_outcome_metrics import analyze, distance

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT=ROOT/'evals_action_success5_20261009'
MANIFEST=ROOT/'robocasa_data/val.json'
BUNDLES=ROOT/'evals_diagnostics_20261009'
TEXT='/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base'
KINDS=('original','random_xyz','flip_gripper')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,obj):
    p=Path(path);p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def frozen(path,obj):
    if path.exists():
        if json.loads(path.read_text())!=obj:raise ValueError('Existing protocol differs: '+str(path))
    else:write(path,obj)

def encode(rgb,path):
    h,w=rgb.shape[1:3];tmp=path.with_name(path.stem+'.partial.mp4')
    subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','5','-i','pipe:0',
                    '-an','-c:v','libx264','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],input=rgb.tobytes(),capture_output=True,check=True)
    tmp.replace(path)
    meta=json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(path)],capture_output=True,check=True).stdout)
    assert int(meta['streams'][0]['nb_read_frames'])==len(rgb)

def decode(path):
    raw=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(-1,256,320,3)

def prepare():
    rows=read_manifest(MANIFEST,'robocasa')['cases'];chosen=[];groups=set()
    for row in sorted(rows,key=lambda r:(r['initial_state_id'],r['id'])):
        if row['success'] and row['initial_state_id'] not in groups and row['frames']-1>END:
            chosen.append(row);groups.add(row['initial_state_id'])
            if len(chosen)==5:break
    assert len(chosen)==5 and len(groups)==5
    protocol={'manifest':str(MANIFEST),'manifest_sha256':sha(MANIFEST),'cases':chosen,
              'selection':'first GT-success episode in each of first five different validation initial-state groups; not selected by model quality',
              'actions':{'seed':SEED,'modified_rows_start_inclusive_end_exclusive':[START,END],
                         'variants':list(KINDS),'implementation_sha256':sha(ROOT/'action_intervention_new_codec.py'),
                         'note':'Exact existing intervention: smoothed random XYZ offset clipped to [-1,1], or gripper dim6 sign flip, four13D control blocks per action row. Same intervention direction across cases.'},
              'beam':4,'memory':'full','initial_real_frames':2,'prediction_seed':0,
              'bundles':{m:{'path':str(BUNDLES/(m+'_bundle')),'manifest_sha256':sha(BUNDLES/(m+'_bundle')/'manifest.json')} for m in ('success','natural')},
              'limitation':'Only original actions have a ground-truth future. Action sensitivity does not establish counterfactual physical correctness.'}
    frozen(OUT/'selection.json',protocol)
    for row in chosen:
        _,actions,_=read_case(MANIFEST,row,52);modified,offset=interventions(actions)
        for name,a in modified.items():
            assert a.shape==actions.shape and np.isfinite(a).all()
            assert np.array_equal(a[:START],actions[:START]) and np.array_equal(a[END:],actions[END:])
            if name!='original':assert np.any(a!=actions)
    print(json.dumps({'selected':[r['id'] for r in chosen],'true_successes':5,'models':2,'videos':30,'offset':offset.tolist()},indent=2),flush=True)

def panel(real,movies,name,path):
    sources=[real]+[movies[k] for k in KINDS];labels=['REAL (original action)',name+' ORIGINAL',name+' RANDOM XYZ',name+' FLIP GRIPPER']
    frames=[]
    for t in range(len(real)):
        im=Image.new('RGB',(1280,288),'black');d=ImageDraw.Draw(im)
        for i,(rgb,label) in enumerate(zip(sources,labels)):
            im.paste(Image.fromarray(rgb[t]),(320*i,32));d.text((320*i+5,4),label,fill='white')
            d.text((320*i+5,17),f'target={t+2}'+(' MODIFIED ACTION' if START-1<=t<END-1 and i>=2 else ''),fill='yellow')
        frames.append(np.asarray(im))
    encode(np.stack(frames),path);Image.fromarray(frames[-1]).save(path.with_suffix('.png'))

def summarize():
    selection=json.loads((OUT/'selection.json').read_text());metrics=[]
    for m in ('success','natural'):
        for row in selection['cases']:
            p=OUT/row['id']/m/'metrics.json'
            if p.exists():metrics.append(json.loads(p.read_text()))
    write(OUT/'summary.json',{'completed_model_cases':len(metrics),'target_model_cases':10,'models':metrics})
    lines=['# 五条真实成功轨迹上的动作扰动','','两个模型：自己的success-only与success+failure 10k checkpoint，沿用其本地适配codec。5条来自5个验证初态组，不按预测质量挑选。每条原动作/随机XYZ/反转夹爪各一次beam4，共30条预测。',
           '',f'已完成 {len(metrics)}/10 个模型×案例。','',
           '只改旧实验的样本选择，扰动规则保持一致：action行[8,26)，每行四个13维控制块；XYZ加固定种子的平滑随机偏移并裁剪到[-1,1]；或夹爪第6维取反。只给最初两帧真实图像，随后完整episode自回归。输入改动最早影响真实target9，即预测序列下标7。',
           '','|案例|模型|原动作末帧距真实px|原动作末帧距目标px|XYZ首个变化预测帧|夹爪首个变化预测帧|','|---|---|---:|---:|---:|---:|']
    def fmt(v):return '未检出' if v is None else f'{v:.2f}'
    for m in metrics:
        ds=m['outcomes']['original']['distances']
        lines.append(f"|{m['id']}|{m['model']}|{fmt(ds['real'])}|{fmt(ds['goal'])}|{m['comparisons']['random_xyz']['first_changed_prediction_frame']}|{m['comparisons']['flip_gripper']['first_changed_prediction_frame']}|")
    lines+=['','首个变化帧从0计数，0对应真实target2；指标使用未压缩预测帧和动态token。原动作先核查与GT的差异，真实成功不保证WM原动作预测正确。末帧红色不可见单列，不当成失败。',
            '','## 动作变化引起的预测变化','',
            '|案例|模型|扰动|改变的action token数|扰动生效前相同|生效后图像MAE 0–255|末帧方块相对原预测位移px|',
            '|---|---|---|---:|---|---:|---:|']
    for m in metrics:
        for k,r in m['comparisons'].items():
            lines.append(f"|{m['id']}|{m['model']}|{k}|{m['changed_action_tokens'][k]}|{r['prefix_unchanged']}|{r['mean_mae255_after']}|{fmt(r['last_cube_shift_vs_original'])}|")
    lines+=['','## 如何看视频','',
            '每条案例目录下 `comparison_success.mp4` 和 `comparison_natural.mp4`：从左到右为真实成功视频、原动作预测、XYZ扰动、夹爪反转。`comparison_both.mp4` 上排仅成功模型、下排混合模型。对应PNG是末帧；各模型目录还保留独立MP4和metrics.json。',
            '','扰动后的动作没有物理仿真真值，因此不把对原GT的偏离当作反事实预测错误，也不根据画面发生改变就声称遵循了正确物理规律。大幅扰动可能离开训练动作分布，本次沿用原参数供探索性比较。',
            '','原720条E1在本实验期间暂缓，结束后自动从已保存进度恢复。']
    (OUT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')

def run():
    protocol=json.loads((OUT/'selection.json').read_text());assert sha(MANIFEST)==protocol['manifest_sha256'];torch.set_num_threads(2)
    for name in ('success','natural'):
        bundle=BUNDLES/(name+'_bundle');assert sha(bundle/'manifest.json')==protocol['bundles'][name]['manifest_sha256']
        pipe=WorldPipeline(bundle,'robocasa','sft','cuda',TEXT);pipe.model.eval()
        for row in protocol['cases']:
            dest=OUT/row['id']/name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'metrics.json').exists():continue
            started=time.time();images,actions,instruction=read_case(MANIFEST,row,52);horizon=len(images)-2
            variants,offset=interventions(actions)
            tokenized={k:encode_actions(torch.from_numpy(a).cuda(),pipe.ranges,52).cpu().numpy() for k,a in variants.items()}
            counts={k:int(np.count_nonzero(t!=tokenized['original'])) for k,t in tokenized.items() if k!='original'}
            assert all(n>0 for n in counts.values())
            for k,t in tokenized.items():
                assert np.array_equal(t[:START],tokenized['original'][:START]) and np.array_equal(t[END:],tokenized['original'][END:])
            predictions={};tokens={};movies={};outcomes={}
            for kind,a in variants.items():
                torch.manual_seed(0)
                with torch.inference_mode():result=pipe.predict(images[:2],a,instruction,horizon,4,'full')
                predictions[kind]=result['prediction'];tokens[kind]=result['tokens'].reshape(horizon,80)
                movies[kind]=np.rint(result['prediction']*255).clip(0,255).astype(np.uint8)
                encode(movies[kind],dest/(kind+'.mp4'));Image.fromarray(movies[kind][-1]).save(dest/(kind+'_last.png'))
                outcomes[kind]=analyze(movies[kind],images)
                print(json.dumps({'event':'prediction_complete','model':name,'id':row['id'],'variant':kind,'frames':horizon,'elapsed':round(time.time()-started,1)}),flush=True)
            original=predictions['original'];comparisons={}
            for kind in ('random_xyz','flip_gripper'):
                delta=np.abs(predictions[kind]-original)*255;per_frame=delta.mean(axis=(1,2,3));changed=(tokens[kind]!=tokens['original']).sum(-1)
                assert not np.any(changed[:START-1]),'Prediction tokens changed before modified action'
                assert delta[:START-1].max()<1e-4,'RGB changed before modified action'
                first=next((i for i,v in enumerate(changed) if v>0),None)
                shifts=[distance(a['xy'],b['xy']) for a,b in zip(outcomes[kind]['tracking'],outcomes['original']['tracking'])]
                comparisons[kind]={'first_changed_prediction_frame':first,'first_changed_real_target':first+2 if first is not None else None,
                                   'prefix_unchanged':True,'mean_mae255_after':float(per_frame[START-1:].mean()),'last_mae255':float(per_frame[-1]),
                                   'last_cube_shift_vs_original':shifts[-1],'per_frame_cube_shift_vs_original':shifts,
                                   'per_frame_mae255':per_frame.tolist(),'per_frame_changed_dynamic_tokens':changed.tolist()}
            record={'id':row['id'],'initial_state_id':row['initial_state_id'],'true_success':True,'model':name,'horizon':horizon,'offset':offset.tolist(),
                    'changed_action_tokens':counts,'comparisons':comparisons,'outcomes':outcomes,'seconds':time.time()-started}
            panel(images[2:],movies,name,OUT/row['id']/('comparison_'+name+'.mp4'))
            write(dest/'metrics.json',record)
            write(dest/'action_inputs.json',{k:a.tolist() for k,a in variants.items()})
            summarize();print('case_complete',name,row['id'],flush=True)
            del predictions,tokens,movies,outcomes
        del pipe;gc.collect();torch.cuda.empty_cache()
    for row in protocol['cases']:
        dest=OUT/row['id'];top=dest/'comparison_success.mp4';bottom=dest/'comparison_natural.mp4';path=dest/'comparison_both.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(top),'-i',str(bottom),'-filter_complex','[0:v][1:v]vstack=inputs=2[v]','-map','[v]','-c:v','libx264','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],capture_output=True,check=True)
        a=Image.open(top.with_suffix('.png'));b=Image.open(bottom.with_suffix('.png'));im=Image.new('RGB',(1280,576));im.paste(a,(0,0));im.paste(b,(0,288));im.save(path.with_suffix('.png'))
    summarize();write(OUT/'complete.json',{'state':'COMPLETE','episodes':5,'models':2,'generated_videos':30,'script_sha256':sha(Path(__file__))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','summarize']);p.add_argument('--resume-e1-after',action='store_true');args=p.parse_args();OUT.mkdir(exist_ok=True)
    try:globals()[args.phase]()
    finally:
        if args.resume_e1_after:subprocess.run(['/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python',str(ROOT/'start_expanded_e1.py')],check=True)
