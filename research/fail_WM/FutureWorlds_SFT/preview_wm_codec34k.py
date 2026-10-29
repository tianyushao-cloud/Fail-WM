#!/usr/bin/env python3
"""Four fixed validation cases, two new 10k WMs, original-action beam4 videos."""
import gc,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from futureworlds.data import read_case
from futureworlds.exporting import export_model
from futureworlds.pipeline import WorldPipeline
from futureworlds.checkpoints import file_sha256
from action_success_five import encode,decode,write,frozen

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT=ROOT/'previews_wm_codec34k_20261012'
MANIFEST=ROOT/'robocasa_data/val.json'
TEXT='/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base'
OUT.mkdir(exist_ok=True)
rows={r['id']:r for r in json.loads(MANIFEST.read_text())['cases']}
ids=['g00006_b00','g00006_b03','g00006_b04','g00014_b06']
models={}
for name in ('success','natural'):
    cfg=json.loads((ROOT/'experiments_mixed_codec_wm_20261011'/f'{name}.json').read_text())
    ck=Path(cfg['output'])/'checkpoints/step_010000'
    assert json.loads((Path(cfg['output'])/'complete.json').read_text())['state']=='COMPLETE'
    dest=OUT/f'{name}_bundle';export_model(cfg['bundle'],'robocasa',ck,dest)
    models[name]={'bundle':str(dest),'checkpoint':str(ck),'checkpoint_receipt_sha256':file_sha256(ck/'experiment.json'),
                  'codec_sha256':file_sha256(dest/'robocasa/codec/diffusion_pytorch_model.safetensors')}
assert models['success']['codec_sha256']==models['natural']['codec_sha256']=='d0d9cf1e8e8d335f2f2a08f6cff4a7cba007d5bb3419bfba8e74c6baf0c35d59'
protocol={'cases':[rows[k] for k in ids],'models':models,'manifest_sha256':file_sha256(MANIFEST),
          'selection':'Fixed before generation: one successful grasp/place, one missed grasp, one early release, one old-codec false-lift case',
          'initial_GT_frames':2,'actions':'original unchanged actions','horizon':'entire remaining episode','beam':4,'memory':'full','seed':0,
          'labels':{'success':'SUCCESS_ONLY: 443 success','natural':'MIXED: 443 success + 357 failure'},
          'no_judge':'Exploratory videos only, no success-rate estimate, no new 720-rollout evaluation'}
frozen(OUT/'selection.json',protocol)
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory)
for name,spec in models.items():
    pipe=WorldPipeline(spec['bundle'],'robocasa','sft','cuda',TEXT)
    for ident in ids:
        dest=OUT/ident;dest.mkdir(exist_ok=True)
        if (dest/f'{name}_complete.json').exists():continue
        images,actions,text=read_case(MANIFEST,rows[ident],52);start=time.monotonic()
        torch.manual_seed(0)
        with torch.inference_mode():result=pipe.predict(images[:2],actions,text,len(images)-2,4,'full')
        movie=np.rint(result['prediction']*255).clip(0,255).astype(np.uint8)
        assert movie.shape==images[2:].shape
        encode(movie,dest/f'{name}.mp4')
        if not (dest/'real.mp4').exists():encode(images[2:],dest/'real.mp4')
        # Save uncompressed frames for visual review; do not retain large RGB NPZ.
        times=sorted(set([2,len(images)-1]+list(range(10,len(images),8))))
        for t in times:Image.fromarray(movie[t-2]).save(dest/f'{name}_t{t:03d}.png')
        np.save(dest/f'{name}_tokens.npy',result['tokens'],allow_pickle=False)
        write(dest/f'{name}_complete.json',dict(model=name,id=ident,frames=len(movie),seconds=time.monotonic()-start,
                                              mp4_sha256=file_sha256(dest/f'{name}.mp4'),review_frames=times))
        print(name,ident,len(movie),'frames',round(time.monotonic()-start,1),'seconds',flush=True)
    del pipe;gc.collect();torch.cuda.empty_cache()
for ident in ids:
    dest=OUT/ident;images,_,_=read_case(MANIFEST,rows[ident],52)
    movies=[images[2:],decode(dest/'success.mp4'),decode(dest/'natural.mp4')]
    titles=['GROUND TRUTH','SUCCESS_ONLY (443 success)','MIXED (443 success + 357 failure)']
    frames=[]
    for i in range(len(images)-2):
        im=Image.new('RGB',(960,296),'black');draw=ImageDraw.Draw(im)
        for col,(movie,title) in enumerate(zip(movies,titles)):
            im.paste(Image.fromarray(movie[i]),(col*320,40));draw.text((col*320+5,4),title,fill='white')
            draw.text((col*320+5,20),f'{ident} t={i+2} | codec34k WM10k',fill='yellow')
        frames.append(np.asarray(im))
    encode(np.stack(frames),dest/'对照_左真实_中仅成功训练_右混合训练.mp4')
    ts=json.loads((dest/'success_complete.json').read_text())['review_frames']
    sheet=Image.new('RGB',(960,296*len(ts)))
    for i,t in enumerate(ts):sheet.paste(Image.fromarray(frames[t-2]),(0,i*296))
    sheet.save(dest/'comparison_keyframes.jpg',quality=94)
    Image.fromarray(frames[-1]).save(dest/'comparison_last.png')
(OUT/'说明.txt').write_text('4条固定验证案例，两组新WM：仅成功443条；混合443成功+357失败。两者使用混合codec step034000，WM step010000。\n每条目录中的对照MP4：左真实，中仅成功训练，右混合训练。各列模型名称指训练数据，不是此次预测结果。\n只输入最初2帧真实画面，原动作不修改，beam4生成完整后续轨迹。展示从第2帧起，不含输入的两帧。5fps方便观看。\n案例：g00006_b00成功；g00006_b03抓取失败；g00006_b04提前释放；g00014_b06失败/旧codec曾虚构抬升。\n本次仅为预览，4案例不能用来估计总体成功率或乐观率。\n')
write(OUT/'complete.json',dict(state='COMPLETE',cases=4,models=2,predictions=8,comparisons=4))
print('ALL PREVIEWS COMPLETE',flush=True)
