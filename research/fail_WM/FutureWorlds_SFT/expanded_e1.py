#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from spatial_outcome_metrics import analyze, references, VERSION

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT=ROOT/'evals_e1_expanded_20261009'
OLD=ROOT/'evals_diagnostics_20261009'
MANIFEST=ROOT/'robocasa_data/val.json'
TEXT='/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base'

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def rows():return json.loads(MANIFEST.read_text())['cases']
def real_images(row):
    with np.load(MANIFEST.parent/row['file'],allow_pickle=False) as z:return z['images'].copy()

def decode(path):
    out=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True)
    return np.frombuffer(out.stdout,dtype=np.uint8).reshape(-1,256,320,3)

def encode(frames,path):
    temp=path.with_name(path.stem+'.partial.mp4')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','320x256','-r','5','-i','pipe:0','-an','-c:v','libx264','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],input=frames.tobytes(),capture_output=True,check=True)
    temp.replace(path)

def select():
    grouped=defaultdict(list)
    for row in rows():
        r=dict(row);r['geometry']=references(real_images(row));grouped[row['initial_state_id']].append(r)
    chosen=[]
    for group, rr in sorted(grouped.items()):
        ss=sorted((r for r in rr if r['success']),key=lambda r:r['id'])[:2]
        ff=[r for r in rr if not r['success']]
        def quality(r):
            g=r['geometry'];sep=min(g['real_to_start'] or 0,g['real_to_goal'] or 0)
            return (sep>=8,r['failure_type'] in ('dropped_before_goal','object_off_counter'),sep)
        ff=sorted(ff,key=lambda r:r['id'])
        ff.sort(key=quality,reverse=True)
        selected=[]
        if ff:
            selected.append(ff[0])
            remaining=[r for r in ff[1:] if r['failure_type']!=ff[0]['failure_type']]
            selected+=(remaining or ff[1:])[:1]
        if len(ss)!=2 or len(selected)!=2:raise ValueError('Need two successes and failures per group')
        chosen+=ss+selected
    artifact={'schema':1,'metrics_version':VERSION,'manifest_sha256':sha(MANIFEST),
              'selection':'2 success and 2 failure per initial group; failures prefer endpoint >=8px from both start and goal, then dropped/off-counter, separation; second failure prefers another type; before generated outputs',
              'cases':chosen,'groups':len(grouped),'counts':dict(Counter('success' if r['success'] else 'failure' for r in chosen)),
              'failure_three_way_identifiable':sum(not r['success'] and min(r['geometry']['real_to_start'] or 0,r['geometry']['real_to_goal'] or 0)>=8 for r in chosen)}
    p=OUT/'selection.json'
    if p.exists() and json.loads(p.read_text())!=artifact:raise ValueError('Frozen selection differs')
    write(p,artifact);print(json.dumps({k:v for k,v in artifact.items() if k!='cases'},indent=2))

def old_audit():
    lookup={r['id']:r for r in rows()}; records=[]
    for case in json.loads((OLD/'e1_summary.json').read_text())['cases']:
        r=lookup[case['id']];real=real_images(r)
        for pred in case['predictions']:
            source=OLD/'e1'/case['model']/case['id']/(pred['name']+'.mp4')
            records.append({'model':case['model'],'id':r['id'],'name':pred['name'],'source':str(source),
                            'old_proxy_outcome':pred['proxy_outcome'],**analyze(decode(source),real)})
    write(OUT/'old_e1_spatial_audit.json',records)
    for model in ('success','natural'):
        for cid in sorted({r['id'] for r in records}):
            rr=[r for r in records if r['model']==model and r['id']==cid]
            print(model,cid,'beam',rr[0]['class'],'samples',Counter(r['class'] for r in rr[1:]),flush=True)

def save_prediction(dest,name,movie,real,metadata):
    path=dest/(name+'.mp4');encode(movie,path)
    raw=analyze(movie,real)
    decoded=analyze(decode(path),real)
    Image.fromarray(movie[-1]).save(dest/(name+'_last.png'))
    record={**metadata,'name':name,'raw':raw,'decoded':decoded,'mp4_sha256':sha(path)}
    write(dest/(name+'.json'),record)
    return record

def case_report(dest):
    records=[json.loads(p.read_text()) for p in [dest/'beam4.json']+[dest/f'sample_{i:02d}.json' for i in range(8)] if p.exists()]
    sheet=Image.new('RGB',(960,3*304),'black');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        x,y=i%3*320,i//3*304;im=Image.open(dest/(r['name']+'_last.png'));sheet.paste(im,(x,y+48))
        s=r['raw'];ds=s['distances'];dist=' '.join(k[0]+'='+('NA' if v is None else f'{v:.1f}') for k,v in ds.items())
        draw.text((x+4,y+3),r['name']+' '+s['class'],fill='white');draw.text((x+4,y+20),dist,fill='white')
    sheet.save(dest/'last_frames.png')

def run(args):
    import torch
    from futureworlds.pipeline import WorldPipeline
    from futureworlds.memory import rollout
    from futureworlds.batch import TensorBatch
    from futureworlds.data import read_case
    selected=json.loads((OUT/'selection.json').read_text())
    if selected['manifest_sha256']!=sha(MANIFEST):raise ValueError('Validation manifest changed')
    torch.set_num_threads(2)
    models={x.split('=',1)[0]:Path(x.split('=',1)[1]) for x in args.model} if args.model else {m:OLD/(m+'_bundle') for m in ('success','natural')}
    receipt={'models':{m:{'bundle':str(p),'manifest_sha256':sha(p/'manifest.json')} for m,p in models.items()},'selection_sha256':sha(OUT/'selection.json'),
             'temperature':1.0,'samples':8,'batch':args.sample_batch,'seed':2026100900,'metrics_version':VERSION}
    receipt_path=OUT/('run_'+('_'.join(models))+'.json')
    if receipt_path.exists() and json.loads(receipt_path.read_text())!=receipt:raise ValueError('Run protocol changed')
    write(receipt_path,receipt)
    cases=selected['cases'][:args.limit] if args.limit else selected['cases']
    for model,bundle in models.items():
        missing=[r for r in cases if not (OUT/'videos'/model/r['id']/'complete.json').exists()]
        if not missing:continue
        pipe=WorldPipeline(bundle,'robocasa','sft',args.device,TEXT);pipe.model.eval()
        for case in missing:
            started=time.time();dest=OUT/'videos'/model/case['id'];dest.mkdir(parents=True,exist_ok=True)
            images,actions,text=read_case(MANIFEST,case,52);horizon=len(images)-2
            metadata={'model':model,'id':case['id'],'success':case['success'],'failure_type':case['failure_type'],'initial_state_id':case['initial_state_id']}
            if not (dest/'real.mp4').exists():encode(images[2:],dest/'real.mp4')
            Image.fromarray(images[-1]).save(dest/'real_last.png')
            if not (dest/'beam4.json').exists():
                with torch.inference_mode():movie=np.rint(pipe.predict(images[:2],actions,text,horizon,4,'full')['prediction']*255).astype(np.uint8)
                save_prediction(dest,'beam4',movie,images,metadata);del movie
            # Encode exactly as training; do not reuse a cache from a different codec.
            with torch.inference_mode():
                ctx,anchor=pipe.encode_observations(images[:2]);_,_,acts,_,_=pipe.encode_training_window(images,actions,text);pipe.condition(text)
            for begin in range(0,8,args.sample_batch):
                count=min(args.sample_batch,8-begin)
                names=[f'sample_{i:02d}' for i in range(begin,begin+count)]
                if all((dest/(name+'.json')).exists() for name in names):continue
                # Same seeds across models; case-specific to avoid shared random streams across episodes.
                seed=2026100900+int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16)+begin
                gen=torch.Generator(device=args.device).manual_seed(seed)
                with torch.inference_mode():
                    frames,_=rollout(pipe.model,ctx.reshape(1,-1)+4375,anchor,acts,horizon,count,gen,temperature=1.0)
                    codes=torch.stack([x['response'] for x in frames],1)
                    pix=pipe.worker.detokenize(TensorBatch({'tokens':codes,'ctx_tokens':ctx.expand(count,-1,-1)}),None).batch['pixels'][:,1:]
                    movies=np.rint(pix.permute(0,1,3,4,2).cpu().numpy()*255).astype(np.uint8)
                for name,movie in zip(names,movies):save_prediction(dest,name,movie,images,{**metadata,'seed':seed,'sample_batch':count})
                del frames,codes,pix,movies
            case_report(dest);write(dest/'complete.json',{'model':model,'id':case['id'],'predictions':9,'seconds':time.time()-started})
            print(json.dumps({'event':'case_complete','model':model,'id':case['id'],'seconds':round(time.time()-started,1),'completed_cases':len(list((OUT/'videos').glob('*/*/complete.json')))},ensure_ascii=False),flush=True)
            report()
        del pipe;gc.collect();torch.cuda.empty_cache()
    report()
    expected=len(models)*len(cases)
    if sum((OUT/'videos'/m/r['id']/'complete.json').exists() for m in models for r in cases)!=expected:raise ValueError('Incomplete evaluation')
    write(OUT/('complete_'+('_'.join(models))+'.json'),{'cases':expected,'predictions':expected*9,'protocol':receipt})

def report():
    records=[]
    for path in sorted((OUT/'videos').glob('*/*/*.json')):
        if path.name=='complete.json':continue
        r=json.loads(path.read_text());records.append(r)
    summary=[]
    for model in sorted({r['model'] for r in records}):
        for success in (True,False):
            for method in ('beam4','sampling'):
                rr=[r for r in records if r['model']==model and r['success']==success and (r['name']=='beam4')==(method=='beam4')]
                if not rr:continue
                for domain in ('raw','decoded'):
                    s=[r[domain] for r in rr]; eligible=[r for r in s if r['goalward_eligible']]
                    summary.append({'model':model,'true_success':success,'method':method,'domain':domain,'videos':len(rr),
                                    'cases':len({r['id'] for r in rr}),'classes':dict(Counter(r['class'] for r in s)),
                                    'ambiguous':sum(r['ambiguous'] for r in s),'far_from_all':sum(r['far_from_all'] is True for r in s),
                                    'last_missing':sum(r['last_missing'] for r in s),'stationary':sum(r['stationary_prediction_proxy'] is True for r in s),
                                    'goalward_eligible':len(eligible),'goalward_missing':sum(r['last_missing'] for r in eligible),
                                    'goalward_counts':{str(m):sum(r['goalward_by_margin'][str(m)] is True for r in eligible) for m in (0,2,4)},
                                    'goalward_rate_bounds':{str(m):[sum(r['goalward_by_margin'][str(m)] is True for r in eligible)/len(eligible),
                                                              (sum(r['goalward_by_margin'][str(m)] is True for r in eligible)+sum(r['last_missing'] for r in eligible))/len(eligible)] if eligible else None for m in (0,2,4)}})
    by_group=[]
    for model in sorted({r['model'] for r in records}):
        for method in ('beam4','sampling'):
            for group in sorted({r['initial_state_id'] for r in records}):
                rr=[r for r in records if r['model']==model and r['initial_state_id']==group and not r['success'] and (r['name']=='beam4')==(method=='beam4') and r['raw']['goalward_eligible']]
                if rr:by_group.append({'model':model,'method':method,'group':group,'videos':len(rr),
                                      'goalward_rate_margin2':sum(r['raw']['goalward_by_margin']['2'] is True for r in rr)/len(rr),
                                      'missing_rate':sum(r['raw']['last_missing'] for r in rr)/len(rr)})
    write(OUT/'summary.json',{'metrics_version':VERSION,'completed_cases':len(list((OUT/'videos').glob('*/*/complete.json'))),'total_predictions':len(records),'groups':summary,'initial_group_rates':by_group})
    lines=['# E1 扩展空间评估（进度随案例完成更新）','','不再用距离目标 5.78px 的二分类推断是否预测对失败。表中使用未压缩生成帧；JSON 同时给出 MP4 解码后的结果。',
           '',f'已完成 {len(records)} 条预测，{len(list((OUT/"videos").glob("*/*/complete.json")))} 个模型×案例。目标为 2 模型×40 案例×9=720 条。','',
           '|模型|GT|解码|视频数|近真实|近目标|近起点|未检出|参照/距离歧义|离所有参照>10px|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['domain']!='raw':continue
        c=r['classes'];lines.append(f"|{r['model']}|{'成功' if r['true_success'] else '失败'}|{r['method']}|{r['videos']}|{c.get('real_like',0)}|{c.get('goal_like',0)}|{c.get('start_like',0)}|{r['last_missing']}|{r['ambiguous']}|{r['far_from_all']}|")
    lines+=['','四类是末帧最近参照点的空间标签，不是物理成功分类。真实位置与目标/起点可能重合；最近两参照相距<5px 或预测到它们距离差<2px 时标歧义。离所有参照>10px 单列，不能称为正确预测。未检出仅指红色检测器未找到足够像素，可能是遮挡、物体出界或图像退化。','',
            '|模型|解码|失败且可识别视频数|偏向目标（margin 0/2/4px）|末帧未检出|margin2代理比例/含缺失上界|', '|---|---|---:|---|---:|---|']
    for r in summary:
        if r['domain']!='raw' or r['true_success']:continue
        n=r['goalward_eligible'];vals='/'.join(str(r['goalward_counts'][str(m)]) for m in (0,2,4));bounds=r['goalward_rate_bounds']['2'];rate='/'.join(f'{v:.1%}' for v in bounds) if bounds else 'NA';lines.append(f"|{r['model']}|{r['method']}|{n}|{vals}|{r['goalward_missing']}|{rate}|")
    lines+=['','“偏向目标”代理：真实失败末态与目标、起点与目标均至少8px，且预测离目标比离真实位置和起点都近（报告0/2/4px margin敏感性）。分母包含末帧不可见；可见子集分母=可识别数−未检出数。可识别失败样本的总体乐观率代理下界=偏目标数/分母，上界=(偏目标数+未检出数)/分母；这不是经人工验证的真实成功率。',
            '','成功控制用于识别整体不搬动物体的模型。每条 JSON 额外保存全程方块轨迹、相对初态最大位移、相对首个预测帧最大位移（<6px且至少80%帧可见才暂记静止）、与真实轨迹误差。仅靠末态近起点不能证明“没有动过”。',
            '','案例及优先选择理由见 selection.json；原36条视频重新审计见 old_e1_spatial_audit.json。视频和无损末帧在 videos/{model}/{case}/，九宫格 last_frames.png。模型间、同组分支及8次采样不是独立场景，不按720个独立样本计算显著性。',
            '','新模型可复用固定 selection.json：expanded_e1.py run --model 标签=/绝对路径/bundle。默认使用旧两个10k模型。']
    (OUT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','old_audit','run','report']);p.add_argument('--model',action='append');p.add_argument('--limit',type=int);p.add_argument('--sample-batch',type=int,choices=[1,2,4,8],default=4);p.add_argument('--device',default='cuda');a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.phase=='run':run(a)
    else:globals()[a.phase]()
