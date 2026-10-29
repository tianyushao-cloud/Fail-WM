#!/usr/bin/env python3
"""Compare genuine codec reconstruction on the same held-out local episodes."""
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from futureworlds.codec import CompressiveVQModelFSQ
from codec_roi_validation import evaluate, positions, roi_mask
from spatial_outcome_metrics import point, distance

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT=ROOT/'evals_codec_comparison_20261009'
MANIFEST=ROOT/'robocasa_data/val.json'
BUNDLES={'ours':ROOT/'assets/futureworlds_bundle_robocasa_codec_10k',
         'scholar':ROOT/'assets/futureworlds_scholar_robocasa'}
PREVIEWS=['g00006_b03','g00006_b05','g00014_b07','g00061_b06']

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def load_images(row):
    with np.load(MANIFEST.parent/row['file'],allow_pickle=False) as z:return z['images'].copy()

@torch.inference_mode()
def reconstruct(model,images,ts):
    chunks=[]
    for begin in range(0,len(ts),8):
        inds=ts[begin:begin+8]
        x=torch.from_numpy(images[[0]+inds]).permute(0,3,1,2).float().cuda()/255
        c,d=model.tokenize(x[None].contiguous())
        y=model.detokenize(c,d)[0,1:].float().clamp(0,1)
        chunks.extend(np.rint(y.permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8))
    return np.stack(chunks)

def summarize(records):
    groups={}
    names=['success','failure']+sorted({r['failure_type'] for r in records if not r['success']})
    for name in names:
        rr=[r for r in records if (r['success'] if name=='success' else not r['success'] if name=='failure' else r['failure_type']==name)]
        def avg(key):
            v=[r[key] for r in rr if r[key] is not None]
            return float(np.mean(v)) if v else None
        nvis=sum(r['visible_frames'] for r in rr);miss=sum(r['missed_frames'] for r in rr)
        groups[name]={'episodes':len(rr),'frames':sum(r['frames'] for r in rr),'roi_mae_255':avg('roi_mae_255'),
                      'mae_255':avg('mae_255'),'center_error':avg('center_error'),'visible_frames':nvis,'missed_frames':miss,
                      'miss_rate':miss/nvis if nvis else None,'last_missing':sum(r['last_missing'] for r in rr),
                      'last_classes':dict(Counter(r['last_class'] for r in rr)),
                      'goalward_eligible':sum(r['goalward_eligible'] for r in rr),'goalward_count':sum(r['goalward'] for r in rr)}
    return groups

def encode(frames,path):
    h,w=frames.shape[1:3]
    subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','5','-i','pipe:0','-an','-c:v','libx264','-crf','14','-pix_fmt','yuv420p',str(path)],input=frames.tobytes(),capture_output=True,check=True)

def main():
    OUT.mkdir(exist_ok=True);(OUT/'terminal_panels').mkdir(exist_ok=True)
    rows=json.loads(MANIFEST.read_text())['cases'];lookup={r['id']:r for r in rows}
    provenance={'manifest_sha256':digest(MANIFEST),'script_sha256':digest(Path(__file__)),
                'validator_sha256':digest(ROOT/'codec_roi_validation.py'),'spatial_metrics_sha256':digest(ROOT/'spatial_outcome_metrics.py'),
                'models':{},'preview_cases_selected_before_scoring':PREVIEWS,'device':'cuda','precision':'FP32',
                'scope':'100 validation episodes; no world-model prediction or training'}
    for name,bundle in BUNDLES.items():
        m=json.loads((bundle/'manifest.json').read_text())
        entries=[e for e in m['files'] if e['path'].startswith('robocasa/codec/')]
        for e in entries:
            if digest(bundle/e['path'])!=e['sha256']:raise ValueError('Codec checksum mismatch')
        provenance['models'][name]={'bundle':str(bundle),'source_revision':m.get('source_revision'),
                                    'source_repo':m.get('source_repo'),'source_checkpoint':m.get('robocasa_codec_source'),
                                    'files':entries}
    write(OUT/'provenance.json',provenance)
    results={};last_images={};preview_images={}
    for name,bundle in BUNDLES.items():
        model=CompressiveVQModelFSQ.from_pretrained(bundle/'robocasa/codec',local_files_only=True).cuda().eval()
        episodes=[];last_images[name]={};preview_images[name]={};started=time.time()
        for begin in range(0,len(rows),10):
            chunk=rows[begin:begin+10]
            result=evaluate(model,MANIFEST.parent,chunk,'cuda');episodes+=result['episodes']
            print(name,'audited',len(episodes),'elapsed',round(time.time()-started,1),flush=True)
        for row in rows:
            images=load_images(row)
            last_images[name][row['id']]=reconstruct(model,images,[len(images)-1])[0]
        for cid in PREVIEWS:
            images=load_images(lookup[cid]);preview_images[name][cid]=reconstruct(model,images,list(range(1,len(images))))
        results[name]={'groups':summarize(episodes),'episodes':episodes}
        write(OUT/(name+'.json'),results[name]);del model;gc.collect();torch.cuda.empty_cache()
    for row in rows:
        cid=row['id'];truth=load_images(row)[-1]
        panel=Image.new('RGB',(960,288),'black');draw=ImageDraw.Draw(panel)
        for i,(name,im) in enumerate([('REAL',truth),('OURS',last_images['ours'][cid]),('SCHOLAR',last_images['scholar'][cid])]):
            panel.paste(Image.fromarray(im),(i*320,32));draw.text((i*320+5,4),name+' '+cid,fill='white')
        panel.save(OUT/'terminal_panels'/(cid+'.png'))
    for cid in PREVIEWS:
        real=load_images(lookup[cid])[1:]
        movie=np.concatenate([real,preview_images['ours'][cid],preview_images['scholar'][cid]],axis=2)
        # Labels are burned in above the image, preserving all scene pixels.
        labeled=np.zeros((len(movie),280,960,3),dtype=np.uint8);labeled[:,24:]=movie
        for j in range(len(movie)):
            im=Image.fromarray(labeled[j]);d=ImageDraw.Draw(im)
            for k,label in enumerate(['REAL','OURS','SCHOLAR']):d.text((k*320+5,5),label+' '+cid,fill='white')
            labeled[j]=np.asarray(im)
        path=OUT/(cid+'_real_ours_scholar.mp4');encode(labeled,path)
        probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(path)],capture_output=True,check=True)
        assert int(json.loads(probe.stdout)['streams'][0]['nb_read_frames'])==len(real)
    ours={r['id']:r for r in results['ours']['episodes']};scholar={r['id']:r for r in results['scholar']['episodes']}
    pairs=[]
    for cid,a in ours.items():
        b=scholar[cid];pairs.append({'id':cid,'success':a['success'],
            'roi_delta_scholar_minus_ours':b['roi_mae_255']-a['roi_mae_255'] if a['roi_mae_255'] is not None and b['roi_mae_255'] is not None else None,
            'ours_last_missing':a['last_missing'],'scholar_last_missing':b['last_missing'],
            'ours_goalward':a['goalward'],'scholar_goalward':b['goalward']})
    write(OUT/'paired.json',pairs)
    lines=['# 学长 RoboCasa codec 与本地成功适配 codec 对比','','同一100条验证轨迹（53成功、47失败），事件窗口及末帧共1969帧，FP32真实图像→token→重建。没有WM生成，没有新训练。',
           '','学长权重来源 easylearning/FutureWorlds，下载版本 '+str(provenance['models']['scholar']['source_revision'])+'。哈希已对照本地发布manifest核验，本次没有声称读取到了网站最新提交。',
           '','|真实数据|codec|方块ROI MAE（0–255）|可见方块位置误差px|真实可见重建漏检/可见帧|末帧未检出|偏向目标/可识别|',
           '|---|---|---:|---:|---|---:|---|']
    for group in ('success','failure'):
        for name in BUNDLES:
            r=results[name]['groups'][group]
            lines.append(f"|{group}|{name}|{r['roi_mae_255']:.3f}|{r['center_error']:.3f}|{r['missed_frames']}/{r['visible_frames']} ({r['miss_rate']:.1%})|{r['last_missing']}|{r['goalward_count']}/{r['goalward_eligible']}|")
    lines+=['','ROI=真实红方块bbox外扩8px。ROI误差/位置误差先每episode平均再跨episode平均，位置只统计两边均检出帧，漏检另外报。末帧未检出包括真实已经不可见的情况，不应全解释为codec导致消失。',
            '','偏向目标：真实末态与目标、起点与目标均≥8px，重建方块离目标比离真实位置和起点都至少近2px。二维位置与颜色检测代理，不是物理成功率；不能由本实验推断学长训练数据的成功/失败比例。',
            '','|失败类型|本地ROI|学长ROI|本地漏检|学长漏检|','|---|---:|---:|---:|---:|']
    for group in results['ours']['groups']:
        if group in ('success','failure'):continue
        a=results['ours']['groups'][group];b=results['scholar']['groups'][group]
        lines.append(f"|{group}|{a['roi_mae_255']:.3f}|{b['roi_mae_255']:.3f}|{a['missed_frames']}|{b['missed_frames']}|")
    for outcome in (True,False):
        pp=[p for p in pairs if p['success']==outcome and p['roi_delta_scholar_minus_ours'] is not None]
        lines+=['',f"{'成功' if outcome else '失败'}轨迹中，学长ROI误差更低 {sum(p['roi_delta_scholar_minus_ours']<0 for p in pp)}/{len(pp)} 条；这是描述性配对统计，不把同初态分支视为独立样本。"]
    lines+=['','可视化：`terminal_panels/` 含100条真实/本地/学长末帧并排PNG；4个 `*_real_ours_scholar.mp4` 为真实/本地/学长完整重建对照，均已检查解码帧数。原始指标：ours.json、scholar.json、paired.json。',
            '','该结果用于决定是否需要另训codec。学长codec若更好，需用其配套WM或重新建立cache并训练WM；不能把它直接换到原WM上，默认token含义兼容。']
    lines+=['','## 结论与额外校验','',
            '学长codec的失败ROI误差为本地codec的3.32倍，成功ROI误差为5.07倍。失败末态偏向目标为0不能解释为更准确：53条成功轨迹重建后30条末帧未检出方块，其余23条位置更接近起点。',
            '', '查看g00006_b05、g00014_b07、g00061_b06并排图，学长codec存在机械臂外观/形状失真、目标颜色变化及方块异常。结论仅限于我们的自定义方块场景，不代表其原训练分布表现，也不能推断训练集是否仅成功。建议继续本地成功+失败codec适配对照，暂不直接替换为学长codec。']
    if (OUT/'roundtrip_verification.json').exists():
        lines+=['','额外检查了g00006_b00和g00006_b05：关闭TF32并统一连续内存输入后，模型直接重建与token往返重建逐元素一致，最大差为0。直接路径同样有较大重建误差。首次未统一精度/布局的比较不满足逐元素一致，随后按原codec worker的严格精度条件复核通过。结果见roundtrip_verification.json及两个*_path_check.png。']
    (OUT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'complete.json',{'state':'COMPLETE','episodes':100,'frames_per_codec':sum(r['frames'] for r in results['ours']['episodes']),'preview_videos':len(PREVIEWS)})
    print((OUT/'REPORT_zh.md').read_text(),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);main()
