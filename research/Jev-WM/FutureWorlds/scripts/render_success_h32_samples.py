"""Render existing SFT/RL test predictions without inference or reward changes."""
import json,hashlib,subprocess,html
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
ROOT=Path('/media/tyshao/Jev-WM/FutureWorlds')
RUN=ROOT/'runs/local_success_memspo_h32'
OUT=RUN/'sample_review';OUT.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def index(label):
 d={}
 for f in (RUN/'evaluation/test'/label).glob('case_*/result.json'):
  r=load(f);d[r['id']]=(f.parent,r)
 return d
a=index('sft');b=index('rl');assert set(a)==set(b)
rows=[]
for id in a:
 sa=a[id][1]['metrics']['32'];sb=b[id][1]['metrics']['32']
 rows.append(dict(id=id,sft=sa,rl=sb,delta=sb['lpips']-sa['lpips'],percent=100*(sb['lpips']/sa['lpips']-1)))
chosen=[];used=set()
for category,ordered,n in [('improved',sorted(rows,key=lambda r:r['delta']),3),
                           ('regressed',sorted(rows,key=lambda r:-r['delta']),3),
                           ('similar',sorted(rows,key=lambda r:abs(r['delta'])),2)]:
 groups=set()
 for r in ordered:
  group=r['id'].split('_b')[0]
  if group in groups or r['id'] in used:continue
  chosen.append(dict(r,category=category));used.add(r['id']);groups.add(group)
  if len(groups)==n:break
write=lambda p,v:p.write_text(json.dumps(v,indent=2,ensure_ascii=False))
write(OUT/'selection.json',dict(checkpoint_step=load(RUN/'checkpoint_selection.json')['selected_step'],split='test',
 selection='Largest absolute LPIPS gains/losses and smallest changes; distinct initial states within category. Illustrative selected cases, not representative success rates.',
 playback='8 frames/s for viewing, not asserted capture rate',cases=chosen))
for row in chosen:
 id=row['id'];folder=OUT/(row['category']+'__'+id);folder.mkdir(exist_ok=True)
 with np.load(a[id][0]/'prediction.npz') as za,np.load(b[id][0]/'prediction.npz') as zb:
  assert np.array_equal(za['truth'],zb['truth']) and np.array_equal(za['observations'],zb['observations'])
  ref=za['truth'];sft=np.rint(np.clip(za['prediction'],0,1)*255).astype(np.uint8);rl=np.rint(np.clip(zb['prediction'],0,1)*255).astype(np.uint8)
  initial=za['observations'].copy()
  assert ref.shape==sft.shape==rl.shape==(32,256,320,3)
 v=np.concatenate([initial,ref]).astype(np.int16)
 mask=(v[...,0]>65)&(v[...,0]>1.6*v[...,1])&(v[...,0]>1.6*v[...,2])
 yy,xx=np.where(mask.any(axis=0))
 box=[max(0,int(xx.min())-48),max(0,int(yy.min())-72),min(320,int(xx.max())+49),min(256,int(yy.max())+49)] if len(xx) else [0,0,320,256]
 frames=[]
 for t in range(32):
  canvas=Image.new('RGB',(960,600),'#eeeeee');draw=ImageDraw.Draw(canvas)
  for col,(label,seq) in enumerate([('GROUND TRUTH',ref),('SFT step10000',sft),('RL step100',rl)]):
   x=col*320;im=Image.fromarray(seq[t])
   draw.text((x+8,8),f'{label} | frame {t+2}',fill='black')
   canvas.paste(im,(x,30))
   detail=ImageOps.contain(im.crop(box),(320,270),Image.Resampling.NEAREST)
   canvas.paste(detail,(x+(320-detail.width)//2,310))
   draw.text((x+8,290),'Same reference-defined crop',fill='black')
  draw.text((8,583),id+' | original observations: frames 0/1 | playback 8 fps',fill='black')
  frames.append(canvas)
 cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x600','-r','8','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'comparison.mp4')]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 for f in frames:proc.stdin.write(f.tobytes())
 proc.stdin.close()
 if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
 frames[0].save(folder/'comparison.gif',save_all=True,append_images=frames[1:],duration=125,loop=0)
 # Four sheets include every generated frame, full scene and enlarged common crop.
 for block in range(4):
  sheet=Image.new('RGB',(1280,720),'white');draw=ImageDraw.Draw(sheet)
  for rr,(label,seq) in enumerate([('GT',ref),('SFT',sft),('RL100',rl)]):
   for col in range(8):
    t=block*8+col;x=col*160;y=rr*240
    im=Image.fromarray(seq[t]);sheet.paste(im.resize((160,128)),(x,y+18))
    detail=ImageOps.contain(im.crop(box),(160,90),Image.Resampling.NEAREST)
    sheet.paste(detail,(x,y+148));draw.text((x+3,y+3),f'{label} frame {t+2}',fill='black')
  sheet.save(folder/f'frames_{block}.jpg',quality=93)
 # Poster focuses on largest per-frame LPIPS change (diagnostic choice, disclosed).
 delta=np.array(b[id][1]['per_frame']['lpips'])-np.array(a[id][1]['per_frame']['lpips'])
 t=int(np.argmax(np.abs(delta)));frames[t].save(folder/'poster.jpg',quality=95)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(folder/'comparison.mp4')]))
 assert probe['streams'][0]['nb_read_frames']=='32'
 row['folder']=folder.name
 write(folder/'metadata.json',dict(**row,crop_box=box,original_frames=[0,1],predicted_frames=[2,33],poster_frame=t+2,
  poster_rule='maximum absolute per-frame LPIPS difference',sft_source=str(a[id][0]),rl_source=str(b[id][0]),
  source_sha256=dict(sft=digest(a[id][0]/'prediction.npz'),rl=digest(b[id][0]/'prediction.npz')),video_check=probe))
 print(json.dumps(dict(id=id,category=row['category'],percent=row['percent'],video=str(folder/'comparison.mp4'))),flush=True)
labels={'improved':'指标改善','regressed':'指标退化','similar':'指标接近'}
cards=[]
for r in chosen:
 f=r['folder']
 links=' '.join(f'<a href="{f}/frames_{i}.jpg">第{i*8+2}–{i*8+9}帧</a>' for i in range(4))
 cards.append(f'<article><h2>{labels[r["category"]]} · {html.escape(r["id"])}</h2><p>32帧 LPIPS：SFT {r["sft"]["lpips"]:.5f} → RL {r["rl"]["lpips"]:.5f}（{r["percent"]:+.1f}%）</p><video controls loop muted preload="none" poster="{f}/poster.jpg" src="{f}/comparison.mp4"></video><p><a href="{f}/comparison.gif">GIF</a> · {links}</p></article>')
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>成功数据 RL 与 SFT 样本对照</title><style>body{max-width:1080px;margin:30px auto;padding:0 18px;font:17px system-ui;background:#f5f5f5;color:#222}article{background:white;padding:20px;margin:25px 0;border-radius:12px}video{width:100%}a{margin-right:12px}h2{font-size:20px}</style><h1>成功数据 RL 与 SFT：8例对照</h1><p>左：真实画面；中：SFT；右：RL 第100步。下半部为同一局部裁切。均从原始首两帧预测后续32帧，8 fps仅为播放速度。</p><p>按测试集LPIPS挑选3例改善、3例退化、2例接近；不是随机代表样本，也不能将LPIPS当作行为正确率。观察方块复制/消失、夹爪接触、搬运路径与释放时序；真实画面是参照。封面选取该例逐帧LPIPS变化最大的时刻，完整视频保留全部32帧。</p>'''+''.join(cards)+'</html>'
(OUT/'index.html').write_text(page)
lines=['# 成功数据 RL 样本对照','','结论：测试集均值小幅改善，不保证每条轨迹改善。以下是按指标选择的诊断例，不能用于估算行为成功率。','',
'打开 index.html 可播放全部对照。每例 comparison.mp4 / comparison.gif 左真实、中SFT、右RL100；下半部是统一局部放大。frames_0..3.jpg 覆盖全部32帧。',
'','|类别|样本|SFT LPIPS|RL LPIPS|相对变化|','|---|---|---:|---:|---:|']
for r in chosen:lines.append(f'|{labels[r["category"]]}|{r["id"]}|{r["sft"]["lpips"]:.5f}|{r["rl"]["lpips"]:.5f}|{r["percent"]:+.1f}%|')
lines+=['','三个类别内尽量使用不同初始场景；这些样本来自原59例成功测试集，未重新推理或重新训练。','所有文件保存在服务器。']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
write(OUT/'complete.json',dict(state='COMPLETE',cases=len(chosen),videos_verified_32_frames=True))

