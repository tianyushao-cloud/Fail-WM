from pathlib import Path
import json,subprocess,hashlib,zipfile,textwrap,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'rt1_jev_h32_probe_20261017'
OUT=ROOT/'review';OUT.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
data=json.loads((ROOT/'results.json').read_text());cards=[]
for scene in data['results']:
    cursor=scene['scene']-1
    with np.load(BASE/f'generated/{cursor:02d}/frames.npz') as z:sft=z['predictions'].copy();gt=z['reference'].copy();initial=z['initial'].copy()
    with np.load(ROOT/f'generated/{cursor:02d}/frames.npz') as z:rl=z['predictions'].copy();assert np.array_equal(gt,z['reference'])
    for candidate in range(4):
        name=f'scene{cursor+1}_group{candidate}';folder=OUT/name;folder.mkdir(exist_ok=True)
        def render(t):
            canvas=Image.new('RGB',(960,344),(18,23,32));draw=ImageDraw.Draw(canvas)
            for i,line in enumerate(textwrap.wrap(scene['task'],100)[:2]):draw.text((8,4+i*19),line,font=font,fill='white')
            for col,(label,clip) in enumerate([('GT',gt),('SFT',sft[candidate]),('R0 RL step200',rl[candidate])]):
                frame=initial[t] if t<2 else clip[t-2]
                draw.text((col*320+8,46),label,font=font,fill=(122,212,255));canvas.paste(Image.fromarray(frame),(col*320,70))
            draw.text((8,326),f'Scene {cursor+1}, beam group {candidate} | t={t} | display 4fps, not original timing',font=font,fill='white')
            return canvas
        frames=[render(t) for t in range(34)]
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x344','-r','4','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'comparison.mp4')],input=b''.join(np.asarray(f).tobytes() for f in frames),check=True)
        sheet=Image.new('RGB',(960,344*4))
        for row,t in enumerate([2,11,21,33]):sheet.paste(frames[t],(0,row*344))
        sheet.save(folder/'keyframes.png')
        s,r=scene['SFT'][candidate],scene['RL'][candidate]
        cards.append(dict(scene=cursor+1,group=candidate,task=scene['task'],video=name+'/comparison.mp4',sheet=name+'/keyframes.png',sft=s,rl=r))
summary=json.loads((ROOT/'summary.json').read_text())
header='''<!doctype html><meta charset="utf-8"><title>SFT 与 R0 RL：32帧停滞对照</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f3f6f8;color:#183044}main{max-width:1200px;margin:auto}.box{background:white;padding:18px;border-radius:10px;margin:18px 0}video{width:100%}select,button{font:inherit;padding:6px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}a{color:#166995}pre{white-space:pre-wrap}</style><main><h1>SFT 与 R0 RL：32帧停滞对照</h1><div class="box">相同3个起点、动作序列和搜索配置，各4个候选。所有视频都收录，无挑选。R0 RL是已训练200步、训练预测10帧的权重；本次只进行32帧推理。<p>这3个窗口属于RL训练集，只用于现象诊断，不能作为泛化提升证据。候选编号代表beam组，不代表两模型预测了同一条轨迹。重复帧按原始RGB逐字节相等计算。</p><pre id="summary"></pre></div><div class="box"><select id="select"></select><h2 id="task"></h2><video id="video" controls loop></video><p>左GT / 中SFT / 右R0 RL。含2帧初始+32帧预测；4fps仅为展示速度。</p><p id="stats"></p><a id="sheet" target="_blank">打开无损关键帧对照</a></div><div class="box"><h2>全部候选统计</h2><table><thead><tr><th>场景</th><th>beam组</th><th>SFT相邻重复/31</th><th>RL相邻重复/31</th><th>SFT最长连续不变</th><th>RL最长连续不变</th></tr></thead><tbody id="rows"></tbody></table><p>“连续不变”包含该段首帧，例如17帧完全相同对应16次相邻重复。运动增加不等于动作或交互更正确，需结合GT看视频。</p><a href="results.json">完整统计</a> · <a href="protocol.json">协议</a></div></main>'''
script='''<script>const cards=__CARDS__,summary=__SUMMARY__;const $=id=>document.getElementById(id);$('summary').textContent=JSON.stringify(summary,null,2);cards.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent='场景'+c.scene+' · beam组'+c.group;$('select').append(o);let tr=document.createElement('tr');for(let v of [c.scene,c.group,c.sft.adjacent_repeats,c.rl.adjacent_repeats,c.sft.longest_constant_run.length,c.rl.longest_constant_run.length]){let td=document.createElement('td');td.textContent=v;tr.append(td)}$('rows').append(tr)});function show(){let c=cards[$('select').value];$('task').textContent=c.task;$('video').src=c.video;$('sheet').href=c.sheet;$('stats').textContent='相邻重复：SFT '+c.sft.adjacent_repeats+'/31 → RL '+c.rl.adjacent_repeats+'/31；最长连续不变：SFT '+c.sft.longest_constant_run.length+'帧 → RL '+c.rl.longest_constant_run.length+'帧。'}$('select').onchange=show;show();</script>'''
(OUT/'index.html').write_text(header+script.replace('__CARDS__',json.dumps(cards,ensure_ascii=False).replace('<','\\u003c')).replace('__SUMMARY__',json.dumps(summary)))
for name in ['results.json','summary.json','protocol.json']:(OUT/name).write_bytes((ROOT/name).read_bytes())
for f in OUT.glob('*/comparison.mp4'):
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(f)],capture_output=True,text=True,check=True)
    assert int(json.loads(result.stdout)['streams'][0]['nb_frames'])==34
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
archive=ROOT/'SFT_vs_R0_RL_32frames.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,'SFT_vs_R0_RL_32frames/'+str(p.relative_to(OUT)))
print(json.dumps(dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive=str(archive))))
