from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'rt1_jev_segment_calibration_20261018'
summary=json.loads((ROOT/'summary.json').read_text())
cases=json.loads((ROOT/'cases.json').read_text())
receipts=[json.loads(p.read_text()) for p in (ROOT/'requests').glob('*/receipt.json')]
current={(r['identity']['case']['id'],r['identity']['segment'],r['identity']['layout'],r['identity']['repeat']):r for r in receipts}
old={(r['identity']['case']['id'],r['identity']['order'],r['identity']['repeat']):r for p in (OLD/'requests').glob('*/receipt.json') for r in [json.loads(p.read_text())]}
def frac(x):return dict(numerator=sum(x),denominator=len(x),fraction=sum(x)/len(x) if x else None)
matched={};dims=['structure','dynamics','interaction']
for version in ['v1','v2']:
    agree=[];stable=[]
    for c in cases:
        if c['cohort']!='old_train_diagnostic':continue
        for s in range(4):
            if version=='v1':a=old[c['id'],'reference_first',0];b=old[c['id'],'prediction_first',0];rep=old.get((c['id'],'reference_first',1))
            else:a=current.get((c['id'],s,'reference_left',0));b=current.get((c['id'],s,'prediction_left',0));rep=current.get((c['id'],s,'reference_left',1))
            if not a or not b:continue
            for d in dims:
                name=f's{s+1}_{d}';agree.append(a['answers'][name]['raw']==b['answers'][name]['raw'])
                aa=[r['answers'][name]['accepted'] for r in [a,b]+([rep] if rep else [])];stable.append(len(set(aa))==1 and aa[0]!='uncertain')
    matched[version]=dict(raw_layout_agreement=frac(agree),stable_confident_cells=frac(stable))
freeze_any=[];freeze_raw=[]
for r in receipts:
    i=r['identity']
    if i['case']['kind']=='freeze' and i['segment']>=2:
        a=r['answers'][f's{i["segment"]+1}_dynamics'];freeze_any.append(a['accepted'] in ['partial','wrong']);freeze_raw.append(a['raw'] in ['partial','wrong'])
artifact=dict(matched_old_cases=matched,freeze_any_deviation_accepted=frac(freeze_any),freeze_any_deviation_raw=frac(freeze_raw),http_attempts=json.loads((ROOT/'http_budget.json').read_text())['attempts'],unique_request_ids=len(set(r['transport']['request_id'] for r in receipts)),validation_informative_pairs=sum(bool(p['winner']) for p in summary['validation_pairs']),validation_pairs=len(summary['validation_pairs']),notes=['Freeze severe-error gate may be too strict when GT changes modestly; do not interpret that gate alone as judge failure.','Token diversity does not imply task-progress diversity.','Two validation windows are now judge-development data, not final evaluation.'])
(ROOT/'audit.json').write_text(json.dumps(artifact,indent=2))
def pretty(x):return f'{x["numerator"]}/{x["denominator"]} ({100*x["fraction"]:.1f}%)' if x['fraction'] is not None else '未完成'
lines=['# Jev 对齐分段校准（v2）','',f'完成 {len(receipts)} 次 API 评分，23个案例。权重未改动，训练更新0步。','',
'## 改动','',
'保留32帧完整联系图作为背景，每次只评一个8帧片段。每个目标时间点以原分辨率并排显示GT和预测，并附上一帧作为边界。左右位置交换；旧RL案例追加同布局重复。类别、三个维度、置信度门槛（P≥0.60，第一二名差≥0.20）沿用v1。','',
'使用2个新验证窗口，每个生成4条RL/ReCAP候选，按既有清单前两条选取；另保留7个旧案例。验证窗口与RL训练episode不重叠，但现在用于裁判开发，不是最终测试。','',
'## 同一批旧案例的直接比较','',
'| 指标 | v1 | v2 |','|---|---:|---:|',
f'| 交换位置/顺序的原始分类一致率 | {pretty(matched["v1"]["raw_layout_agreement"])} | {pretty(matched["v2"]["raw_layout_agreement"])} |',
f'| 跨布局与已有重复均高置信度一致 | {pretty(matched["v1"]["stable_confident_cells"])} | {pretty(matched["v2"]["stable_confident_cells"])} |','',
'## 本轮预设检查','', '| 检查 | 结果 | 通过 |','|---|---:|---:|']
for k,v in summary['metrics'].items():lines.append(f'| {k} | {pretty(v)} | {summary["gates"].get(k,"辅助指标")} |')
lines += ['',f'新验证窗口内的12个候选配对中，严格分数范围不重叠的配对：{artifact["validation_informative_pairs"]}/12。',
'分数范围是对不确定类别枚举所有可能值产生的上下界，不是统计置信区间；结构/动态/交互权重0.2/0.4/0.4，严重结构错误段封顶0.25，严重交互错误段封顶0.5。该汇总规则在本轮请求前固定。',
'', '## 控制样本限制','',
f'冻结后半段若允许“部分不符”也算检测到偏差，则高置信度检测率为 {pretty(artifact["freeze_any_deviation_accepted"])}；此为补充统计，未替换预设严重错误门槛。',
'移动橙子的GT后半段有放下后的撤手过程，冻结帧仍保留部分正确状态；打开抽屉窗口的部分时段运动幅度也不大。预设“冻结后半段都应判wrong”可能过严，不能仅据此认定裁判失效。候选视觉token不同，也不保证可见任务进展有明显优劣。',
'','## 可复核材料','',
'index.html：23条GT/预测视频、每段原始类别和概率、左右交换/重复结果、分数范围。input_examples/index.html：实际发送给API的PNG输入样例（两个布局）。protocol.json与selection.json保留预设规则。responses.json保留原始响应，含request id。',
'','## 验证','',
'目标8帧的原始像素与GT/预测时间索引逐帧核对；交换左右不翻转图像像素；总览包含32帧；封顶函数有直接检查。导出视频验证34帧，文件附SHA256。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
soft=json.loads((ROOT/'soft_diagnostic.json').read_text()) if (ROOT/'soft_diagnostic.json').exists() else None
if soft and soft['completed_requests']==len(receipts):
    with (ROOT/'REPORT.md').open('a') as f:
        f.write('\n## 连续评分的额外分析\n\n用类别概率求等级期望，将uncertain概率质量保留为上下界，并取左右布局及重复请求的包络。该分析在本轮运行中加入，未替换预设门槛。新验证候选12对中，范围分离且差值大于0.05的配对为 '+str(soft['informative_pairs'])+'/12。概率未经校准，不能把范围当成置信区间。\n\n## 结论\n\n对齐输入在相同旧案例中略有改善，但未提供通过预设检查的组内排序信号。本轮未启动正式RL训练。即使放松“冻结必须严重错误”的控制样本假设，真实候选排序信号不足的问题仍在。当前只有一张共享A100可见，也不满足用户要求的四卡正式训练环境。\n\n下一步应优先检验针对GT对齐局部片段的直接候选相对判断，或扩大候选的任务相关差异；不能把未知项硬填成0，也不能据此宣称Jev本身无用。进一步训练前需要在少量新案例上确认排序与人工可见证据一致。\n')
print(json.dumps(artifact,indent=2))
