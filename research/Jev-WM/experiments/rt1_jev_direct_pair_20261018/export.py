from pathlib import Path
import json,subprocess,hashlib,zipfile,textwrap,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Jev_direct_pair';OUT.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
summary=json.loads((ROOT/'summary.json').read_text());receipts=[json.loads(p.read_text()) for p in (ROOT/'requests').glob('*/receipt.json')];cards=[]
diagnostics=json.loads((ROOT/'diagnostics.json').read_text())
for pair in summary['pairs']:
    if not pair['segment_results']:continue
    folder=OUT/pair['id'];folder.mkdir(exist_ok=True)
    with np.load(ROOT/'pairs'/pair['id']/'frames.npz') as z:arrays={k:z[k].copy() for k in z.files}
    def render(t):
        im=Image.new('RGB',(960,344),(18,23,32));d=ImageDraw.Draw(im)
        for i,l in enumerate(textwrap.wrap(pair['task'],100)[:2]):d.text((8,4+i*19),l,font=font,fill='white')
        for col,(name,clip) in enumerate([('REFERENCE',arrays['reference']),(pair['left'],arrays['left']),(pair['right'],arrays['right'])]):
            frame=arrays['initial'][t] if t<2 else clip[t-2]
            d.text((col*320+8,46),name,font=font,fill=(122,212,255));im.paste(Image.fromarray(frame),(col*320,70))
        label=f'Initial {t+1}' if t<2 else f'Future {t-1}/32 | segment {(t-2)//8+1}'
        d.text((8,326),label+' | display 4fps',font=font,fill='white');return im
    frames=[render(t) for t in range(34)]
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x344','-r','4','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'comparison.mp4')],input=b''.join(np.asarray(f).tobytes() for f in frames),check=True)
    sheet=Image.new('RGB',(960,1376))
    for row,t in enumerate([9,17,25,33]):sheet.paste(frames[t],(0,row*344))
    sheet.save(folder/'keyframes.png')
    rs=sorted([r for r in receipts if r['identity']['pair']['id']==pair['id']],key=lambda r:(r['identity']['segment'],r['identity']['repeat'],r['identity']['swap']))
    (folder/'responses.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2))
    cards.append(dict(**pair,video=pair['id']+'/comparison.mp4',sheet=pair['id']+'/keyframes.png',raw=pair['id']+'/responses.json',reviews=[dict(segment=r['identity']['segment']+1,swap=r['identity']['swap'],repeat=r['identity']['repeat'],**r['classification']) for r in rs]))
header='''<!doctype html><meta charset="utf-8"><title>Jev 直接候选比较</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f3f6f8;color:#183044}main{max-width:1200px;margin:auto}.box{background:white;padding:18px;border-radius:10px;margin:18px 0}video{width:100%}select,button{font:inherit;padding:6px;margin:4px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}a{color:#166995}pre{white-space:pre-wrap}</style><main><h1>Jev：对齐局部片段直接比较候选</h1><div class="box">两个验证窗口，各4个RL/ReCAP候选，全部6个配对。GT与A/B按时间戳对齐，每次评一个8帧片段，32帧总览提供上下文。<p>AB/BA都高置信度选择同一实际候选才接受偏好；有重复时也必须一致。C为平局，D或交换/重复冲突均弃权。合成控制为相同视频、GT对后半段冻结、GT对黑屏。</p><p>视频固定为左GT / 中原候选1 / 右原候选2；API会交换A/B映射。含2帧初始+32帧未来，4fps为展示速度。验证窗口已用于裁判开发，不是最终泛化测试。尚未进行训练。</p><p id="status"></p><a href="protocol.json">提示词与预设门槛</a> · <a href="REPORT.md">实验报告</a> · <a href="input_examples/index.html">实际API输入图</a></div><div class="box"><select id="pair"></select><button id="prev">上一对</button><button id="next">下一对</button><h2 id="task"></h2><video id="video" controls loop></video><p id="result"></p><table><thead><tr><th>未来帧段</th><th>AB原始判断</th><th>BA原始判断</th><th>重复判断（若有）</th><th>最终接受</th></tr></thead><tbody id="segments"></tbody></table><p>A/B为当次请求标签；括号为最高类别概率。低于0.60或第一二名差小于0.20时弃权。</p><a id="sheet">无损关键帧</a> · <a id="raw">完整回答和概率</a></div><div class="box"><h2>校准统计</h2><table><thead><tr><th>指标</th><th>结果</th><th>通过</th></tr></thead><tbody id="metrics"></tbody></table></div></main>'''
script='''<script>const cards=__CARDS__,summary=__SUMMARY__,$=x=>document.getElementById(x);const names={identical_tie_fraction:'相同视频判平局',gt_black_correct_fraction:'GT对黑屏选GT',gt_freeze_correct_fraction:'GT对冻结选GT（辅助统计）',gt_freeze_never_prefers_corruption:'GT对冻结未反选冻结',real_physical_order_agreement:'真实配对交换后实际判断一致',repeat_raw_agreement:'相同输入重复一致',real_stable_segment_preference_fraction:'真实片段稳定有效偏好比例'};cards.forEach((p,i)=>{let o=document.createElement('option');o.value=i;o.textContent='窗口'+p.window+' · '+p.left+' vs '+p.right+' · '+p.kind;$('pair').append(o)});function display(v,p){return v==='left'?p.left:v==='right'?p.right:v==='C'?'平局':v==='conflict'?'不同片段胜者冲突':'弃权/未确定'}function show(){let p=cards[Number($('pair').value)];$('task').textContent=p.task;$('video').src=p.video;$('sheet').href=p.sheet;$('raw').href=p.raw;$('result').textContent='整条轨迹：'+display(p.trajectory_decision,p);$('segments').replaceChildren();p.segment_results.forEach(s=>{let tr=document.createElement('tr');const format=r=>r?r.raw+' ('+r.probability.toFixed(2)+')'+(r.accepted!==r.raw?' →弃权':''):'未评';let a=p.reviews.find(r=>r.segment===s.segment&&!r.swap&&!r.repeat),b=p.reviews.find(r=>r.segment===s.segment&&r.swap&&!r.repeat),rep=p.reviews.filter(r=>r.segment===s.segment&&r.repeat);for(let x of [((s.segment-1)*8+1)+'–'+s.segment*8,format(a),format(b),rep.map(format).join(' / ')||'—',display(s.decision,p)]){let td=document.createElement('td');td.textContent=x;tr.append(td)}$('segments').append(tr)})}$('pair').onchange=show;$('prev').onclick=()=>{$('pair').value=(Number($('pair').value)+cards.length-1)%cards.length;show()};$('next').onclick=()=>{$('pair').value=(Number($('pair').value)+1)%cards.length;show()};$('status').textContent='完成6次基本检查+18次诊断，共24次；原144次计划因控制失败停止。校准'+(summary.calibration_passed?'通过':'未通过')+'；训练更新0步。';Object.entries(summary.metrics).forEach(([k,v])=>{let tr=document.createElement('tr');for(let x of [names[k],v.denominator?v.numerator+'/'+v.denominator+' ('+(100*v.fraction).toFixed(1)+'%)':'未评',summary.gates[k]===undefined?'辅助指标':summary.gates[k]?'是':'否']){let td=document.createElement('td');td.textContent=x;tr.append(td)}$('metrics').append(tr)});show();</script>'''
header=header.replace('两个验证窗口，各4个RL/ReCAP候选，全部6个配对。','原计划评两个验证窗口的全部配对，但首轮控制失败后停止扩量。本包只展示实际评过的窗口2 group0/group1、相同视频、GT/黑屏三组，第25–32帧。')
extra='<div class="box"><h2>失败后的18次输入/选项诊断</h2><p>local_only：去掉总览，只输入8张三列图；local_reverse_options：相同局部输入，选项顺序改为D/C/B/A；local_separate_semantic：各时间戳的GT/A/B分开为24张图，换成语义选项（先列平局/不足）。每种变体均测相同视频、GT/黑屏、真实候选，且交换A/B。</p><table><thead><tr><th>变体</th><th>案例</th><th>A实际内容</th><th>B实际内容</th><th>原始回答</th><th>最高概率</th><th>实际选择</th></tr></thead><tbody>'
for r in sorted(diagnostics,key=lambda r:(r['identity']['variant'],r['identity']['pair']['id'],r['identity']['swap'])):
    i=r['identity'];p=i['pair'];a,b=(p['right'],p['left']) if i['swap'] else (p['left'],p['right'])
    ph=p[r['physical']] if r['physical'] in ['left','right'] else r['physical']
    extra+=f'<tr><td>{i["variant"]}</td><td>{p["kind"]}</td><td>{a}</td><td>{b}</td><td>{r["raw"]}</td><td>{r["probability"]:.2f}</td><td>{ph}</td></tr>'
extra+='</tbody></table><a href="diagnostics.json">18次完整API回答</a><p>通过阈值的答案也可能错误；原始类别与是否可信必须分开看。本轮没有足够依据将这些选择作为RL奖励。</p></div>'
(OUT/'index.html').write_text(header.replace('</main>',extra+'</main>')+script.replace('__CARDS__',json.dumps(cards,ensure_ascii=False).replace('<','\\u003c')).replace('__SUMMARY__',json.dumps({k:v for k,v in summary.items() if k!='pairs'})))
for name in ['summary.json','protocol.json','pairs.json','diagnostic_protocol.json','diagnostics.json','REPORT.md']:(OUT/name).write_bytes((ROOT/name).read_bytes())
for name in ['probe.py','diagnose.py']:(OUT/name).write_bytes((ROOT/name).read_bytes())
shutil.copytree(ROOT/'input_examples',OUT/'input_examples',dirs_exist_ok=True)
html='<meta charset="utf-8"><title>实际API输入</title><style>body{font:16px system-ui;margin:24px}img{max-width:1200px;width:100%}</style><h1>窗口2 group0/group1，第25–32帧</h1><p>下面是实际发送给API的PNG，两套输入交换A/B候选，图像本身不镜像。</p>'
for name in ['AB','BA']:
    html+='<details><summary>'+name+'</summary>'
    for p in sorted((OUT/'input_examples'/name).glob('*.png')):html+='<p>'+p.name+'</p><img loading="lazy" src="'+name+'/'+p.name+'">'
    html+='</details>'
(OUT/'input_examples/index.html').write_text(html)
assert len(list(OUT.glob('*/comparison.mp4')))==len(cards)==3
for p in OUT.glob('*/comparison.mp4'):
    r=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(p)],capture_output=True,text=True,check=True);assert int(json.loads(r.stdout)['streams'][0]['nb_frames'])==34
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2));archive=ROOT/'Jev_direct_pair.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,'Jev_direct_pair/'+str(p.relative_to(OUT)))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
