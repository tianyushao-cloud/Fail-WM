from pathlib import Path
import json,subprocess,hashlib,shutil,zipfile,textwrap
import numpy as np
from PIL import Image,ImageFont,ImageDraw
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'review32';OUT.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
def read(p):return json.loads(Path(p).read_text())
cases=[]
for cursor in range(3):
    src=ROOT/f'generated/{cursor:02d}';meta=read(src/'metadata.json')
    dest=OUT/f'scene{cursor+1}';dest.mkdir(exist_ok=True)
    shutil.copy(src/'frames.npz',dest/'original_frames.npz')
    with np.load(src/'frames.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    paths={}
    for name,seq in [('gt',np.concatenate([arrays['initial'],arrays['reference']]))]+[(f'c{i}',np.concatenate([arrays['initial'],arrays['predictions'][i]])) for i in range(4)]:
        folder=dest/name;folder.mkdir(exist_ok=True);paths[name]=[]
        for t,frame in enumerate(seq):
            path=folder/f'{t:02d}.png';Image.fromarray(frame).save(path);paths[name].append(path.relative_to(OUT).as_posix())
    assignments={h:read(next((ROOT/f'reviews_h{h}/assignments').glob(f'{cursor:06d}_*.json'))) for h in [10,32]}
    for j,pair in enumerate(assignments[32]['pairs']):
        a,b=pair['pair'];assert assignments[10]['pairs'][j]['pair']==[a,b]
        folder=dest/f'pair{j+1}';folder.mkdir(exist_ok=True)
        scores={};links={}
        for h in [10,32]:
            ap=assignments[h]['pairs'][j];records=[]
            for order,rid in zip(['AB','BA'],ap['requests']):
                receipt=read(ROOT/f'reviews_h{h}/requests'/rid/'receipt.json')
                (folder/f'h{h}_{order}.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False))
                ans=receipt['response']['answers'][0]
                records.append(dict(raw=ans['choice'],filtered=ap['labels'][len(records)],probabilities={x['value']:x['probability'] for x in ans['probabilities']},confidence=ans['confidence']))
            scores[str(h)]=dict(orders=records,vote=ap['vote'])
        group=ROOT/'reviews_h32/groups'/assignments[32]['group']
        shutil.copy(group/'request.json',folder/'prompt_and_metadata.json')
        def render(t):
            canvas=Image.new('RGB',(960,344),(18,23,32));d=ImageDraw.Draw(canvas)
            for i,line in enumerate(textwrap.wrap(meta['task'],100)[:2]):d.text((8,4+i*19),line,font=font,fill='white')
            for col,(label,key) in enumerate([('GT / REFERENCE','gt'),(f'A / candidate {a}',f'c{a}'),(f'B / candidate {b}',f'c{b}')]):
                d.text((col*320+8,46),label,font=font,fill=(122,212,255));canvas.paste(Image.open(OUT/paths[key][t]),(col*320,70))
            d.text((8,326),f't={t:02d} | '+('Shared initial' if t<2 else f'Future {t-1}/32')+' | display 4 fps, not source timing',font=font,fill='white')
            return canvas
        frames=[render(t) for t in range(34)]
        video=folder/'GT_A_B_32frames.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x344','-r','4','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],input=b''.join(np.asarray(f).tobytes() for f in frames),check=True)
        sheet=Image.new('RGB',(960,344*4))
        for row,t in enumerate([2,11,21,33]):sheet.paste(frames[t],(0,row*344))
        sheet.save(folder/'contact_sheet.png')
        cases.append(dict(title=f'场景{cursor+1} 配对{j+1}',task=meta['task'],pair=[a,b],scores=scores,paths=paths,folder=folder.relative_to(OUT).as_posix(),video=video.relative_to(OUT).as_posix()))
(OUT/'cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False))
shutil.copy(ROOT/'summary.json',OUT/'summary.json');shutil.copy(ROOT/'protocol.json',OUT/'protocol.json')
summary=read(ROOT/'summary.json')
html='''<!doctype html><meta charset="utf-8"><title>Jev 10 / 32 帧比较</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f3f6f8;color:#183044}main{max-width:1400px;margin:auto}.box{background:white;padding:18px;margin:16px 0;border-radius:10px}select,button{font:inherit;padding:6px}video{width:100%;max-width:1100px}.panels{display:flex;gap:8px}.panels div{width:33.33%}img{width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccc}table{border-collapse:collapse;width:100%}a{color:#166995}input{width:60%}pre{white-space:pre-wrap}</style><main><h1>Jev：相同候选的前10帧与完整32帧</h1><div class="box">同样3个起点、SFT和候选配对；32帧只生成一次，再分别截取前10帧与完整32帧评分。没有RL更新。A/B交换后必须仍支持同一视频才产生奖励。<pre id="sum"></pre></div><div class="box"><select id="select"></select><h2 id="task"></h2><video id="video" controls loop></video><p>4fps仅用于查看，不代表数据采集帧率。视频包含2帧初始+32帧预测。</p><input id="frame" type="range" min="0" max="33" value="21"><span id="time"></span><div class="panels"><div>GT<a id="gl" target="_blank"><img id="g"></a></div><div id="ac">A<a id="al" target="_blank"><img id="a"></a></div><div id="bc">B<a id="bl" target="_blank"><img id="b"></a></div></div><p>点击图片查看无损原图。下表第二次评分将上述A/B交换。</p></div><div class="box"><table><thead><tr><th>帧数/顺序</th><th>原始选择</th><th>P(A)</th><th>P(B)</th><th>P(C)</th><th>P(D)</th><th>过滤后</th><th>该对奖励</th></tr></thead><tbody id="scores"></tbody></table><p id="links"></p><p>阈值：最高概率≥0.60，领先次高≥0.20，否则视为D。C=无明显差异，D=证据不足。返回值没有文字解释。双顺序不一致不产生偏好。</p></div></main><script>const cases=__CASES__,summary=__SUMMARY__;const $=id=>document.getElementById(id);let c;for(let i=0;i<cases.length;i++){let o=document.createElement('option');o.value=i;o.textContent=cases[i].title;$('select').append(o)}$('sum').textContent=JSON.stringify(summary,null,2);function frame(){let t=Number($('frame').value);$('time').textContent='t='+t;for(const [id,key]of [['g','gt'],['a','c'+c.pair[0]],['b','c'+c.pair[1]]]){$(id).src=c.paths[key][t];$(id+'l').href=c.paths[key][t]}}function show(){c=cases[$('select').value];$('task').textContent=c.task+' | A=c'+c.pair[0]+', B=c'+c.pair[1];$('video').src=c.video;$('scores').replaceChildren();$('links').replaceChildren();for(const h of ['10','32']){c.scores[h].orders.forEach((o,i)=>{let tr=document.createElement('tr');for(const v of [h+' / '+(i?'交换BA':'首次AB'),o.raw,...['A','B','C','D'].map(k=>o.probabilities[k].toFixed(2)),o.filtered,c.scores[h].vote]){let td=document.createElement('td');td.textContent=v;tr.append(td)}$('scores').append(tr);let a=document.createElement('a');a.href=c.folder+'/h'+h+'_'+(i?'BA':'AB')+'.json';a.textContent=h+'帧'+(i?'BA':'AB')+'原始回执 ';a.target='_blank';$('links').append(a)})}for(const [name,path]of [['提示词','prompt_and_metadata.json'],['关键帧对照','contact_sheet.png']]){let a=document.createElement('a');a.href=c.folder+'/'+path;a.textContent=name+' ';a.target='_blank';$('links').append(a)}frame()}$('select').onchange=show;$('frame').oninput=frame;show();</script>'''
html=html.replace('__CASES__',json.dumps(cases,ensure_ascii=False).replace('<','\\u003c')).replace('__SUMMARY__',json.dumps(summary))
(OUT/'index.html').write_text(html)
(OUT/'README.md').write_text('解压后打开index.html。全部6对，无挑选。32帧候选生成一次，比较其前10帧与完整32帧；与先前单独生成10帧的候选可能不同。原始像素在各scene/original_frames.npz及PNG中。GT只供裁判；未训练。视频播放4fps仅为展示。阈值和提示词与先前保持一致，仅更新长度标记。\n')
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
archive=ROOT/'Jev_h32_comparison.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,'Jev_h32_comparison/'+str(p.relative_to(OUT)))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
