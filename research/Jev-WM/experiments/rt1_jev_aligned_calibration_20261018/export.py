from pathlib import Path
import json,subprocess,hashlib,zipfile,textwrap,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Jev_aligned_calibration';OUT.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
cases=json.loads((ROOT/'cases.json').read_text());cards=[]
labels={'gt':'GT 与自身','freeze':'后16帧冻结','reverse_late':'后16帧倒序','matched_pause':'GT与预测均停顿（合成对照）','sft_full':'SFT 全部历史','sft_recap':'SFT + ReCAP','rl_recap':'R0 RL + ReCAP'}
raw_receipts=[json.loads(p.read_text()) for p in (ROOT/'requests').glob('*/receipt.json')]
receipts=[]
for case in cases:
    for layout,repeat in [('reference_left',0),('prediction_left',0),('reference_left',1)]:
        rr=[r for r in raw_receipts if r['identity']['case']['id']==case['id'] and r['identity']['layout']==layout and r['identity']['repeat']==repeat]
        if not rr:continue
        answers={k:v for r in rr for k,v in r['answers'].items()}
        receipts.append(dict(identity=dict(case=case,order=layout,repeat=repeat),answers=answers,provisional_reward=None,segment_receipts=rr))

for case in cases:
    name=case['id'];folder=OUT/name;folder.mkdir(exist_ok=True)
    with np.load(ROOT/'cases'/name/'frames.npz') as z:initial=z['initial'].copy();gt=z['reference'].copy();pred=z['prediction'].copy()
    def render(t):
        canvas=Image.new('RGB',(640,344),(18,23,32));draw=ImageDraw.Draw(canvas)
        for i,line in enumerate(textwrap.wrap(case['task'],78)[:2]):draw.text((8,4+i*18),line,font=font,fill='white')
        for col,(label,clip) in enumerate([('REFERENCE',gt),('PREDICTION: '+case['kind'],pred)]):
            frame=initial[t] if t<2 else clip[t-2]
            draw.text((col*320+6,46),label,font=font,fill=(122,212,255));canvas.paste(Image.fromarray(frame),(col*320,70))
        label=f'Initial {t+1}' if t<2 else f'Future {t-1}/32 | segment {(t-2)//8+1}'
        draw.text((8,326),label+' | display 4fps',font=font,fill='white')
        return canvas
    frames=[render(t) for t in range(34)]
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x344','-r','4','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'comparison.mp4')],input=b''.join(np.asarray(f).tobytes() for f in frames),check=True)
    sheet=Image.new('RGB',(640,344*4))
    for row,t in enumerate([9,17,25,33]):sheet.paste(frames[t],(0,row*344))
    sheet.save(folder/'keyframes.png')
    rs=sorted([r for r in receipts if r['identity']['case']['id']==name],key=lambda r:(r['identity']['repeat'],r['identity']['order']!='reference_first'))
    (folder/'responses.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2))
    cards.append(dict(**case,label=labels.get(case['kind'], 'RL候选组 '+str(case['group'])),video=name+'/comparison.mp4',sheet=name+'/keyframes.png',raw=name+'/responses.json',reviews=[dict(order=r['identity']['order'],repeat=r['identity']['repeat'],answers=r['answers'],reward=r['provisional_reward']) for r in rs]))
summary=json.loads((ROOT/'summary.json').read_text())
analysis={'bounds':summary['bounds']}
soft=json.loads((ROOT/'soft_diagnostic.json').read_text())
header='''<!doctype html><meta charset="utf-8"><title>Jev 分段评分校准</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f3f6f8;color:#183044}main{max-width:1080px;margin:auto}.box{background:white;padding:18px;border-radius:10px;margin:18px 0}video{width:100%;max-width:900px}select,button{font:inherit;padding:6px;margin:4px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}a{color:#166995}.correct{color:#157448}.wrong{color:#b92626}.partial{color:#a56100}.uncertain{color:#666}pre{white-space:pre-wrap}small{display:block;color:#657383}</style><main><h1>Jev 分段评分校准</h1><div class="box">每次请求只评一个8帧片段；GT/预测逐帧并排，全32帧联系图提供背景。评分维度和置信度门槛沿用第一版。所有请求均使用 gpt-6-luna Decisions API。<p>相同视频交换REFERENCE与PREDICTION左右位置；旧RL案例另做同布局重复。页面视频固定左REFERENCE / 右PREDICTION，API实际布局见下拉框。场景1–3为旧训练窗口，场景4–5为新验证窗口，已用于裁判开发。</p><p><b>本页保留分项判断，用不确定项的全部可能取值计算分数范围，尚未用于训练。</b>不确定或低置信度为弃权，不是0分。错误类型是裁判的分类输出，不是人类确认的诊断。验证窗口与RL训练episode不重叠；本轮用于裁判开发，不作为最终泛化测试。</p><p id="verdict"></p><a href="protocol.json">完整提示词、评分映射与预设门槛</a> · <a href="summary.json">校准统计</a></div><div class="box"><select id="case"></select><button id="prev">上一条</button><button id="next">下一条</button><h2 id="task"></h2><video id="video" controls loop></video><p id="note"></p><select id="review"></select><p id="reward"></p><table><thead><tr><th>未来帧</th><th>结构</th><th>动态</th><th>交互</th><th>主要错误（原始分类）</th></tr></thead><tbody id="scores"></tbody></table><p>单元格展示原始分类及最大概率；不满足 P≥0.60 且第一二名差≥0.20 的结果用于reward时弃权。概率未经经验校准。</p><a id="sheet" target="_blank">无损关键帧</a> · <a id="raw" target="_blank">完整API回答与概率</a></div><div class="box"><h2>校准结果</h2><table><thead><tr><th>检查项</th><th>结果</th><th>通过</th></tr></thead><tbody id="gates"></tbody></table></div></main>'''
script='''<script>const cards=__CARDS__,summary=__SUMMARY__;const $=id=>document.getElementById(id),L={correct:'正确',partial:'部分正确',wrong:'明显错误',uncertain:'无法判断',none:'无明显错误',freeze:'停滞',disappearance:'消失/结构破坏',interaction:'交互不符',motion:'运动/位置不符'},G={positive_controls:'GT与匹配停顿判正确',late_freeze:'冻结后半段判严重动态错误',late_reverse:'倒序后半段识别动态偏差',early_controls:'合成案例未修改前半段正确',real_order_agreement:'真实预测交换左右一致率',repeat_agreement:'同布局重复一致率',stable_real_cells:'跨布局高置信度一致分项',strict_trajectory_coverage:'原严格整轨评分可用率（辅助指标）'};cards.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent='场景 '+c.scene+' · '+c.label;$('case').append(o)});function scores(){let c=cards[Number($('case').value)],r=c.reviews[Number($('review').value)];$('scores').replaceChildren();if(!r){$('reward').textContent='无API回答';return}$('reward').textContent='本次布局的原始分项如下；跨布局的保守分数范围见页面下方，尚未用于训练。';for(let s=1;s<=4;s++){let tr=document.createElement('tr'),td=document.createElement('td');td.textContent=((s-1)*8+1)+'–'+s*8;tr.append(td);for(let d of ['structure','dynamics','interaction','fault']){let a=r.answers['s'+s+'_'+d]||{raw:'uncertain',accepted:'uncertain',probability:0},cell=document.createElement('td');cell.className=a.raw;cell.textContent=L[a.raw];let small=document.createElement('small');small.textContent='P='+a.probability.toFixed(2)+(a.accepted!==a.raw?'；低置信度弃权':'');cell.append(small);tr.append(cell)}$('scores').append(tr)}}function show(){let c=cards[Number($('case').value)];$('task').textContent=c.label+' · '+c.task;$('video').src=c.video;$('sheet').href=c.sheet;$('raw').href=c.raw;$('note').textContent=c.synthetic?'合成控制样本：'+c.label:'原始视频；beam组 '+c.group;$('review').replaceChildren();c.reviews.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(r.order==='reference_left'?'GT在左':'预测在左')+(r.repeat?' · 同顺序重复':'');$('review').append(o)});scores()}$('case').onchange=show;$('review').onchange=scores;$('prev').onclick=()=>{$('case').value=(Number($('case').value)+cards.length-1)%cards.length;show()};$('next').onclick=()=>{$('case').value=(Number($('case').value)+1)%cards.length;show()};$('verdict').textContent='已完成 '+summary.completed_requests+' 次请求；校准'+(summary.calibration_passed?'通过':'未通过')+'；训练更新 '+summary.optimizer_updates+' 步。';Object.entries(summary.metrics).forEach(([k,v])=>{let tr=document.createElement('tr');for(let x of [G[k],v.numerator+'/'+v.denominator+' ('+(v.fraction===null?'NA':(100*v.fraction).toFixed(1)+'%')+')',summary.gates[k]===undefined?'辅助统计':summary.gates[k]?'是':'否']){let td=document.createElement('td');td.textContent=x;tr.append(td)}$('gates').append(tr)});show();</script>'''
extra='<div class="box"><h2>真实预测：不确定项保留后的分数范围</h2><p>这是分类映射下的诊断范围，不是统计置信区间，也未用于训练。仅保留两种展示顺序及已有重复请求中均高置信度一致的分项；其余项枚举所有可能类别，计算最低/最高分。范围重叠时不作排序。本轮计算方法已在请求前固定。</p><table><thead><tr><th>场景</th><th>预测</th><th>最低可能分</th><th>最高可能分</th></tr></thead><tbody>'
for b in analysis['bounds']:extra+=f'<tr><td>{b["scene"]}</td><td>{labels.get(b["kind"],b["kind"])}</td><td>{b["lower"]:.3f}</td><td>{b["upper"]:.3f}</td></tr>'
extra+='</tbody></table><p>高置信度一致才固定该分项；阈值未放宽，不确定不填0。范围不是统计置信区间。</p><a href="summary.json">详细范围与候选排序</a> · <a href="REPORT.md">实验说明</a></div>'
extra+='<div class="box"><h2>连续评分的额外诊断</h2><p>用类别概率计算等级期望，不把正确/部分正确之间的犹豫直接丢弃。无法判断的概率保留为上下界，再覆盖左右布局及重复请求的差异；不是置信区间。本分析未替换预设校准规则，未用于训练。</p><table><thead><tr><th>案例</th><th>最低</th><th>最高</th></tr></thead><tbody>'
for b in soft['records']:extra+=f'<tr><td>{b["id"]}</td><td>{b["lower"]:.3f}</td><td>{b["upper"]:.3f}</td></tr>'
extra+='</tbody></table><p>新验证候选的范围分离且差值超过0.05的配对：'+str(soft['informative_pairs'])+'/12。</p><a href="soft_diagnostic.json">完整分析</a></div>'
(OUT/'index.html').write_text(header.replace('</main>',extra+'</main>')+script.replace('__CARDS__',json.dumps(cards,ensure_ascii=False).replace('<','\\u003c')).replace('__SUMMARY__',json.dumps(summary)))
for name in ['summary.json','protocol.json','cases.json','selection.json','audit.json','soft_rule.json','soft_diagnostic.json','REPORT.md']:(OUT/name).write_bytes((ROOT/name).read_bytes())
for p in OUT.glob('*/comparison.mp4'):
    r=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(p)],capture_output=True,text=True,check=True)
    assert int(json.loads(r.stdout)['streams'][0]['nb_frames'])==34
if (ROOT/'input_examples').exists():
    shutil.copytree(ROOT/'input_examples',OUT/'input_examples',dirs_exist_ok=True)
    example_html='<meta charset="utf-8"><title>实际发送给API的PNG</title><style>body{font:16px system-ui;margin:24px}img{max-width:900px;width:100%}details{margin:20px}</style><h1>实际发送给API的PNG样例</h1><p>每组为初始观测、32帧总览、前一帧和8帧目标片段。不同文件夹交换GT与预测左右位置；全部对应未来17–24帧评分请求。</p>'
    for folder in sorted((OUT/'input_examples').iterdir()):
        if folder.is_dir():
            example_html+='<details><summary>'+folder.name+'</summary>'
            for p in sorted(folder.glob('*.png')):example_html+='<p>'+p.name+'</p><img loading="lazy" src="'+folder.name+'/'+p.name+'">'
            example_html+='</details>'
    (OUT/'input_examples/index.html').write_text(example_html)
    page=OUT/'index.html';page.write_text(page.read_text().replace('<a href="protocol.json">','<a href="input_examples/index.html">查看实际API输入图片</a> · <a href="protocol.json">',1))
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
archive=ROOT/'Jev_aligned_calibration.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,'Jev_aligned_calibration/'+str(p.relative_to(OUT)))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
