"""Package inspection videos, transparent selection, audits, and runnable code."""
from pathlib import Path
import json,zipfile,hashlib,shutil

project=Path(__file__).resolve().parent;root=project/'data/batch1000_failure70'
stats=json.loads((root/'dataset_statistics.json').read_text());audit=json.loads((root/'final_audit.json').read_text())
assert stats['episodes']==1000 and stats['failures']==700 and stats['successes']==300 and audit['all_pass']
runtime=json.loads((project/'data/batch1000/environment_provenance.json').read_text())
runtime['source_files_sha256']={str(p.relative_to(root/'source_code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'source_code').rglob('*')) if p.is_file()}
runtime['dataset_selection']='actual-outcome-conditioned; 3 successful and 7 failed per initial group; original source labels/actions/states/videos unchanged'
(root/'environment_provenance.json').write_text(json.dumps(runtime,indent=2))
(root/'progress.json').write_text(json.dumps({'status':'complete','completed':1000,'requested_episodes':1000,'workers':4,'successes':300,'failures':700,'replay_passed':1000,'alignment_passed':1000,'independent_final_audit_pass':True,'unresolved_errors':[],'actual_failure_types':stats['failure_types']},indent=2))
(root/'statistics.json').write_text(json.dumps(stats,indent=2))
analysis=root/'analysis_tools';analysis.mkdir(exist_ok=True)
for n in ['audit_dataset.py','audit_action_conversion.py','batch_report.py','rebalance_dataset.py','package_failure70.py']:shutil.copy2(project/n,analysis/n)
selection=json.loads((root/'selection_manifest.json').read_text())
readme=f'''# RoboCasa 失败占 70%：1000 条

服务器完整数据：`{root}`。300 条最终成功，700 条最终失败。
{json.dumps(stats['failure_types'],ensure_ascii=False)}
保留所有 100 条原恢复轨迹：93 恢复成功，7 恢复失败。每组至少保留一条参考成功，并尽量保留扰动后仍成功的样本。
每个初态组 3 成功 + 7 失败；100 个初态组按原 split 分配，train/val/test = 800/100/100。

这是一份根据真实最终结果筛选的数据集。原版 1000 条以及所有补采候选均保留。
候选池共 {selection['candidate_pool_episodes']} 条，实际 {selection['candidate_pool_successes']} 成功 / {selection['candidate_pool_failures']} 失败。
完整来源和筛选方法见 selection_manifest.json；全候选及是否被选见 candidate_pool_manifest.jsonl；不改标签、动作、state 或视频。
700/300 是人为控制的数据组成，不是未经筛选的控制策略成功率。比较 world model 时应说明这一选择机制。

解压视频包后打开 inspection/index.html，内含 15 条未经后期修改的成功、失败、恢复样本。
完整 H5 留在服务器（本视频包不包含全部 H5），总数据约 {stats['episode_bytes']/1e9:.2f} GB。
原生 actions/raw 为 T×12；actions/target13 为 T×13。
target13 = 世界 xyz（m）+ 行优先旋转矩阵9D + 夹爪目标（+1打开，-1闭合）。
observations/eef_executed13 是实际执行位姿及开度，单独保存，不能当作预测输入动作。
对齐：RGB[i], state[i], target13[i] -> RGB[i+1], state[i+1]；20 Hz；T actions、T+1 RGB/state。
所有轨迹通过原生动作回放、完整 MP4 解码和同步校验，另有独立 13D 转换与几何审计。

当前为自定义 RoboCasa Kitchen 简单台面任务、Panda 手臂、MuJoCo EGL/OpenGL 固定相机，非官方完整 RoboCasa365/RTX。
事件在20 Hz采样，短暂接触可能遗漏；曾发生 success 事件不代表最终 success。

补采和复现筛选（每轮候选通过四进程真实物理执行）：
```bash
cd /media/tyshao/fail_WM/RoboCasa
.venv/bin/python rebalance_dataset.py plan --round 0
.venv/bin/python batch_collect.py --output data/failure_candidates/round_00 --plan-file data/failure_candidates/round_00_plan.json --workers 4
.venv/bin/python rebalance_dataset.py assemble
# 若某些初态失败候选不足，按输出缺额生成 round 1 / round_01 后再 assemble。
.venv/bin/python batch_report.py --data data/batch1000_failure70
.venv/bin/python audit_action_conversion.py --data data/batch1000_failure70
.venv/bin/python audit_dataset.py --data data/batch1000_failure70 --expected-failures 700
```
安装、字段、控制接口见 README.md；本版采样配比以本说明和 dataset_statistics.json 为准。
'''
(root/'RESULTS_README.md').write_text(readme)
codefiles=list(project.glob('*.py'))+list((project/'rc_collect').glob('*.py'))
codefiles += [project/n for n in ['README.md','BATCH1000_README.md','requirements.txt','requirements-lock.txt','install.sh','provenance.json']]
with zipfile.ZipFile(root/'pipeline_code.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in codefiles:z.write(p,p.relative_to(project))
target=project/'RoboCasa_failure70_inspection.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted((root/'inspection').rglob('*')):
        if p.is_file():z.write(p,'RoboCasa_failure70/'+str(p.relative_to(root)))
    for n in ['dataset_statistics.json','final_audit.json','action_conversion_audit.json','selection_manifest.json','candidate_pool_manifest.jsonl','environment_provenance.json','RESULTS_README.md','pipeline_code.zip','splits.json','plan.json']:z.write(root/n,'RoboCasa_failure70/'+n)
    z.write(project/'README.md','RoboCasa_failure70/README.md')
with zipfile.ZipFile(target) as z:assert z.testzip() is None;assert sum(n.endswith('.mp4') for n in z.namelist())==15
print(json.dumps({'zip':str(target),'bytes':target.stat().st_size,'stats':stats}))
