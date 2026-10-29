import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT=ROOT/'evals_e1_expanded_20261009'
def read(p):return json.loads(p.read_text())
summary=read(OUT/'summary.json');selection=read(OUT/'selection.json')
assert summary['total_predictions']==720 and (OUT/'complete_success_natural.json').exists()
records=[]
for p in sorted((OUT/'videos').glob('*/*/*.json')):
    if p.name!='complete.json':records.append(read(p))
assert len(records)==720
paired={(r['model'],r['id'],r['name']):r for r in records}
three={r['id'] for r in selection['cases'] if not r['success'] and
       r['geometry']['real_to_start'] is not None and r['geometry']['real_to_goal'] is not None and
       min(r['geometry']['real_to_start'],r['geometry']['real_to_goal'])>=8}
lines=['# 720条世界模型评估：供外部模型审阅的完整结果','',
       '这是已经完成的固定验证集诊断，不是物理仿真成功率测试。请重点审阅指标定义、缺失处理、tokenizer混淆和后续实验设计。', '',
       '## 模型及训练背景','',
       '- success / 仅成功：443条成功训练轨迹。natural / 混合：443条成功+357条失败（约44.6%失败），不是后来准备的70%失败版本。',
       '- 两者均从相同RT-1 WM初始化，SFT10000步，有效batch64、每卡batch8，token cache版本；core_lr=1e-5、adapter_lr=1e-4、warmup500，每2000步checkpoint。',
       '- 本次两者共享我们自己仅用443条成功数据适配的codec。实际选用codec是step044000，bundle名称中的10k不能解释为codec只训练了10000步。',
       '- 本次不是学长发布的WM权重，不是新成功+失败codec训练后的WM。新codec的进展不应混进这720条基线结果。', '',
       '## 评估协议','',
       '- validation的10个初态组，各2成功+2失败，共40条。2模型×40案例×(1 beam4+8 sampling)=720条。所有完整episode，仅最初2帧GT输入，后续自回归，无真实未来刷新。',
       '- 失败选择先看GT几何，优先末态离起点和目标均至少8px；随后偏好dropped_before_goal/object_off_counter、较大分离，同组第二例尽量不同失败类型。选择在查看生成结果前冻结。',
       '- sampling temperature=1.0，视觉词表4375，无新增top-k/top-p；每批4条、每案例2批。种子=2026100900+int(SHA256(case_id)前8位hex)+批次开始序号(0或4)，两个模型一致。',
       '- memory为anchor1+recent6，预测每帧80个动态token，动作52维由4个13维控制块组成。分辨率320×256、视频5fps。',
       '- 主结果直接从未压缩生成RGB计算；另外对720条MP4解码检查帧数并重算，summary.json分别保留raw/decoded。统计不是从MP4压缩图像单独推断。', '',
       '## 指标如何定义','',
       '1. 红方块检测：R>75、R>1.5G、R>1.4B，至少8像素；取所有满足条件像素的中心。绿色目标采用固定绿色规则，至少20像素。都是颜色代理，没有经生成视频人工标注校准。',
       '2. 三个参照点：GT末帧方块、GT末帧目标（缺失回退首帧）、GT首帧方块。预测末帧到三个参照距离哪个最小，就标real_like/goal_like/start_like；红色不足8像素则undetected。最近参照即使很远也仍有标签，因此必须结合far_from_all。',
       '3. 歧义附加标记：缺少任一参照，或最近两个参照相距<5px，或到两参照的距离差<2px。far_from_all表示到最近参照也>10px。两者与四类交叉，不是额外互斥类别。',
       '4. goalward（偏向目标）代理：仅限真实失败且GT末态与目标≥8px、GT起点与目标≥8px；要求预测到目标的距离+margin < min(到GT末态距离,到GT起点距离)。报告margin=0/2/4px；主表采用2px。不是离目标<5.78px的旧成功分类。',
       '5. 上述goalward有17个失败案例、136个采样输出（17×8）；三点两两均分离≥8px、能更好区分“真实结果”和“起点”的严格子集只有11例。17例和11例含义不同。',
       '6. 未检出可能来自遮挡、出界、颜色/纹理退化，不能证明消失，也不能算正确失败。报告missing/N及goalward/N；所谓下/上界只是假设缺失输出分别全不偏目标/全偏目标的敏感性区间，不是置信区间，也不是物理乐观率的数学界。',
       '7. 同组分支、同case八次采样并不独立；不把720条当作720个独立样本计算显著性。选择偏向几何可辨失败，也不能代表完整验证/测试集失败分布。', '',
       '## 最终主结果（未压缩RGB，margin2px）','',
       '|模型|解码|可识别失败输出N|偏目标|偏目标/N|未检出|未检出/N|可见输出中偏目标|缺失敏感性区间|',
       '|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in summary['groups']:
    if r['domain']!='raw' or r['true_success']:continue
    n=r['goalward_eligible'];g=r['goalward_counts']['2'];missing=r['goalward_missing']
    lines.append(f"|{'仅成功' if r['model']=='success' else '混合'}|{r['method']}|{n}|{g}|{g/n:.1%}|{missing}|{missing/n:.1%}|{g/(n-missing):.1%}|{g/n:.1%}–{(g+missing)/n:.1%}|")
lines+=['','混合采样的偏目标占全部输出比例从43.4%降至32.4%，但可见输出中的比例仅从59/123=48.0%变为44/96=45.8%。同时未检出从13/136=9.6%增至40/136=29.4%。因此不能把约11个百分点的表面下降全部当作失败预测改善。可见子集本身也有选择偏差。',
        '','beam4两模型主代理都是8/17，但只是总数相同，不代表在同8个案例上出错。没有证据支持“只换采样就已解决乐观”。采样有多种结果不等于真实概率已校准。','',
        '## 四类完整分布（未压缩RGB）','',
        '|模型|GT|解码|N|近真实参照|近目标参照|近起点参照|未检出|歧义（交叉标记）|离所有参照>10px（交叉标记）|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in summary['groups']:
    if r['domain']!='raw':continue
    c=r['classes'];lines.append(f"|{'仅成功' if r['model']=='success' else '混合'}|{'成功' if r['true_success'] else '失败'}|{r['method']}|{r['videos']}|{c.get('real_like',0)}|{c.get('goal_like',0)}|{c.get('start_like',0)}|{r['last_missing']}|{r['ambiguous']}|{r['far_from_all']}|")
lines+=['','成功案例中GT末态与目标相近，real_like/goal_like经常是歧义分类；不能把real_like/N当作成功预测准确率。混合成功采样未检出33/160=20.6%，仅成功8/160=5.0%，说明缺失增加并不限于失败案例。',
        '','严格静止代理（至少80%帧可见，预测方块相对首个可见预测帧最大位移<6px）本次各组均为0；这不能证明模型正确搬运，只表明本条件未检出全程几乎不动的轨迹。末态近起点不能直接称为“没动”。', '',
        '## 11例三点分离子集（补充描述性统计）','',
        '|模型|解码|N|偏目标margin2|未检出|偏目标/N|','|---|---|---:|---:|---:|---:|']
for model in ('success','natural'):
    for method in ('beam4','sampling'):
        rr=[r for r in records if r['model']==model and r['id'] in three and (r['name']=='beam4')==(method=='beam4')]
        g=sum(r['raw']['goalward_by_margin']['2'] is True for r in rr);missing=sum(r['raw']['last_missing'] for r in rr)
        lines.append(f"|{'仅成功' if model=='success' else '混合'}|{method}|{len(rr)}|{g}|{missing}|{g/len(rr):.1%}|")
lines+=['','## 逐配对变化（相同案例及sample编号，不作因果归因）','',
        '状态为goalward（2px）、not_goalward、undetected。两模型使用同种子，但生成历史不同，不是物理反事实真值。','',
        '|解码|仅成功状态|混合状态|输出数|','|---|---|---|---:|']
for method in ('beam4','sampling'):
    transitions=Counter()
    for r in records:
        if r['model']!='success' or r['success'] or not r['raw']['goalward_eligible'] or (r['name']=='beam4')!=(method=='beam4'):continue
        other=paired['natural',r['id'],r['name']]
        def state(x):return 'undetected' if x['last_missing'] else 'goalward' if x['goalward_by_margin']['2'] else 'not_goalward'
        transitions[state(r['raw']),state(other['raw'])]+=1
    for (a,b),n in sorted(transitions.items()):lines.append(f'|{method}|{a}|{b}|{n}|')
lines+=['','## 结论边界及供Opus审阅的问题','',
        '- 支持：两个旧WM在真实失败条件下存在向目标偏移的结果；混合训练在beam4下没有降低本代理总数。',
        '- 尚不支持：混合失败数据完全没用、约47%是真实物理乐观率、采样已解决问题、teacher forcing是主因、tokenizer解释所有错误。',
        '- 两个模型共享仅成功适配codec，这限制了对WM本身的归因；新的混合codec应独立检查，再重新编码cache、按相同WM训练协议对比。',
        '- 本指标没有验证接触/抓取/物体三维轨迹，也没有对720条逐条人工标注；颜色检测器可能把伪红色结构当方块。应抽查目标偏移、歧义、未检出及成功控制。',
        '- 请审阅：如何区分WM乐观与codec退化？可见子集/缺失分布变化是否影响模型比较？是否需要按事件窗口和failure_type拆分，并以初态组为单位统计？E2事件对称采样与codec适配应该如何保持独立对照？', '',
        '## 附件与服务器位置','',
        '服务器根目录：/media/tyshao/fail_WM/FutureWorlds_SFT/evals_e1_expanded_20261009/',
        '- summary.json：raw与decoded、0/2/4px裕量、按初态组比例。',
        '- selection.json：完整40案例、GT几何和数据来源。',
        '- complete_success_natural.json：完成及bundle/selection哈希。',
        '- per_prediction_metrics.jsonl：720条逐输出记录，包括原始/解码末态距离和完整红色轨迹。',
        '- spatial_outcome_metrics.py / expanded_e1.py：评分与生成实现。',
        '- 视频在 videos/{success,natural}/{case}/；本结果包不包含视频或权重。',
        '', '原报告REPORT_zh.md已附带，但其中“上下界”只按上述缺失敏感性含义理解，不能当作物理概率界。']
report=OUT/'RESULTS_FOR_OPUS_zh.md';report.write_text('\n'.join(lines)+'\n')
details=OUT/'per_prediction_metrics.jsonl';details.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))
target=OUT/'WM720评估结果_给Opus.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in [report,details,OUT/'summary.json',OUT/'selection.json',OUT/'complete_success_natural.json',OUT/'REPORT_zh.md',ROOT/'spatial_outcome_metrics.py',ROOT/'expanded_e1.py']:
        z.write(p,p.name)
print('report',report,'zip',target,'bytes',target.stat().st_size,'sha256',hashlib.sha256(target.read_bytes()).hexdigest())
print(report.read_text()[report.read_text().find('## 最终主结果'):report.read_text().find('## 四类')])
