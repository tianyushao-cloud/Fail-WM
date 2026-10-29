from pathlib import Path
import json,subprocess,hashlib,zipfile,textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Full_vs_ReCAP_32frames';OUT.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
results=json.loads((ROOT/'results.json').read_text());assert len(results)==6
cards=[];summary={}
for item in results:
    model=item['model'];scene=item['scene']
    with np.load(ROOT/f'generated/{model}/{scene-1:02d}/frames.npz') as z:
        initial=z['initial'].copy();gt=z['reference'].copy();full=z['full'].copy();recap=z['recap'].copy()
    for group in range(4):
        name=f'{model}_scene{scene}_group{group}';folder=OUT/name;folder.mkdir(exist_ok=True)
        def render(t):
            canvas=Image.new('RGB',(960,344),(18,23,32));draw=ImageDraw.Draw(canvas)
            for i,line in enumerate(textwrap.wrap(item['task'],100)[:2]):draw.text((8,4+i*19),line,font=font,fill='white')
            for col,(label,clip) in enumerate([('GT',gt),('Full history | '+model,full[group]),('ReCAP (anchor1+recent6) | '+model,recap[group])]):
                frame=initial[t] if t<2 else clip[t-2]
                draw.text((col*320+6,46),label,font=font,fill=(122,212,255));canvas.paste(Image.fromarray(frame),(col*320,70))
            draw.text((8,326),f'Scene {scene}, beam group {group} | t={t} | display 4fps, not original timing',font=font,fill='white')
            return canvas
        frames=[render(t) for t in range(34)]
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x344','-r','4','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'comparison.mp4')],input=b''.join(np.asarray(f).tobytes() for f in frames),check=True)
        sheet=Image.new('RGB',(960,344*4))
        for row,t in enumerate([2,11,21,33]):sheet.paste(frames[t],(0,row*344))
        sheet.save(folder/'keyframes.png')
        cards.append(dict(model=model,scene=scene,group=group,task=item['task'],video=name+'/comparison.mp4',sheet=name+'/keyframes.png',full=item['full'][group],recap=item['recap'][group]))
for model in ['SFT','RL']:
    summary[model]={}
    for mode in ['full','recap']:
        ss=[s for x in results if x['model']==model for s in x[mode]]
        summary[model][mode]=dict(adjacent_repeats=sum(s['adjacent_repeats'] for s in ss),transitions=372,clips_with_run_ge10=sum(s['longest_constant_run']>=10 for s in ss),clips=12)
atomic=json.dumps(summary,indent=2);(ROOT/'summary.json').write_text(atomic)
header='''<!doctype html><meta charset="utf-8"><title>全部历史 vs ReCAP：32帧视频对照</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f3f6f8;color:#183044}main{max-width:1200px;margin:auto}.box{background:white;padding:18px;border-radius:10px;margin:18px 0}video{width:100%}select,button{font:inherit;padding:6px;margin-right:8px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}a{color:#166995}pre{white-space:pre-wrap}</style><main><h1>全部历史 vs ReCAP：32帧视频对照</h1><div class="box">相同3个起点、动作和解码配置；SFT、R0 RL step200分别比较。每个场景4个beam组全部展示，共24条对照视频。<p><b>左GT / 中保留全部历史 / 右ReCAP（固定锚点+最近6帧）。</b>含2帧初始观测+32帧未来；4fps仅为展示速度。</p><p>全部历史保留所有状态-动作块；两种方案都保留初始场景，每帧重置KV缓存并重新编号位置。ReCAP复用此前已生成并验证的结果。本次无训练、无Jev调用、无GT刷新。</p><p>这3个窗口来自RL训练集，仅用于诊断。候选编号代表beam组，不代表两方案必然生成相同轨迹。运动增加不等于预测更正确，应结合GT判断。</p></div><div class="box"><select id="model"><option>SFT</option><option>RL</option></select><select id="select"></select><button id="prev">上一条</button><button id="next">下一条</button><h2 id="task"></h2><video id="video" controls loop></video><p id="stats"></p><a id="sheet" target="_blank">打开无损关键帧对照</a></div><div class="box"><h2>全部视频与统计</h2><table><thead><tr><th>权重</th><th>场景/组</th><th>全部历史 重复/31</th><th>ReCAP 重复/31</th><th>全部历史 最长不变</th><th>ReCAP 最长不变</th><th>视频</th></tr></thead><tbody id="rows"></tbody></table><p>重复按未压缩RGB逐字节相等计算；连续不变帧数包含首帧。</p><a href="results.json">详细统计</a> · <a href="summary.json">汇总</a> · <a href="protocol.json">实验协议</a></div></main>'''
script='''<script>const cards=__CARDS__;const $=id=>document.getElementById(id);let selected=[];function show(){const c=selected[Number($('select').value)];$('task').textContent=c.model+' · '+c.task;$('video').src=c.video;$('sheet').href=c.sheet;$('stats').textContent='相邻重复：全部历史 '+c.full.adjacent_repeats+'/31，ReCAP '+c.recap.adjacent_repeats+'/31；最长连续不变：全部历史 '+c.full.longest_constant_run+' 帧，ReCAP '+c.recap.longest_constant_run+' 帧。'}function filter(){selected=cards.filter(c=>c.model===$('model').value);$('select').replaceChildren();selected.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent='场景 '+c.scene+' · beam组 '+c.group;$('select').append(o)});show()}$('model').onchange=filter;$('select').onchange=show;$('prev').onclick=()=>{$('select').value=(Number($('select').value)+selected.length-1)%selected.length;show()};$('next').onclick=()=>{$('select').value=(Number($('select').value)+1)%selected.length;show()};cards.forEach(c=>{let tr=document.createElement('tr');for(let v of [c.model,c.scene+'/'+c.group,c.full.adjacent_repeats,c.recap.adjacent_repeats,c.full.longest_constant_run,c.recap.longest_constant_run]){let td=document.createElement('td');td.textContent=v;tr.append(td)}let td=document.createElement('td'),a=document.createElement('a');a.href=c.video;a.textContent='MP4';td.append(a);tr.append(td);$('rows').append(tr)});filter();</script>'''
(OUT/'index.html').write_text(header+script.replace('__CARDS__',json.dumps(cards,ensure_ascii=False).replace('<','\\u003c')))
for name in ['results.json','summary.json','protocol.json']:(OUT/name).write_bytes((ROOT/name).read_bytes())
videos=list(OUT.glob('*/comparison.mp4'));assert len(videos)==24
for f in videos:
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(f)],capture_output=True,text=True,check=True)
    assert int(json.loads(result.stdout)['streams'][0]['nb_frames'])==34
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
archive=ROOT/'Full_vs_ReCAP_32frames.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,'Full_vs_ReCAP_32frames/'+str(p.relative_to(OUT)))
print(json.dumps(dict(summary=summary,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive=str(archive))))
