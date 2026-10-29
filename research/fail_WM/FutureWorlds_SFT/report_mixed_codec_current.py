#!/usr/bin/env python3
"""Summarize current codec acceptance without changing any original results."""
import json
from pathlib import Path

ROOT = Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT = ROOT / 'evals_codec_current_20261011'
EXP = ROOT / 'experiments_mixed_codec_wm_20261011'
report = json.loads((OUT / 'report.json').read_text())
selected = json.loads((EXP / 'codec_selection.json').read_text())
step = selected['selected']['step']
lines = [f'# 混合数据 codec 验收与 WM 训练准备（2026-10-11）', '',
         f'固定采用混合 codec **step {step}**。训练尚未完成不妨碍使用已保存且验收通过的 checkpoint；不改变仍在训练的 codec 任务。', '',
         '## 生成式解码事件检查', '',
         '固定验证集：24 条从未抬起失败、20 条成功对照、7 条提前释放失败。真实 RGB 经 SFT cache 式分块编码，再按 WM 生成路径两帧一组解码。',
         '显式 FP32，matmul/cudnn TF32 均关闭、benchmark 关闭，与 WorldPipeline 设置一致。此处只评估 codec 重建，尚不是新 WM 预测结果。', '',
         '| Codec | 虚构抬升 /24 | 未抬起组完全漏检 | 成功抬升 /20 | 提前释放落下 /7 | 提前释放漏检 /7 |',
         '|---|---:|---:|---:|---:|---:|']
for label,r in report['codecs'].items():
    s=r['summary']; u=s['unlifted_failure']; c=s['success_control']; rel=s['release_failure']
    lines.append(f"| {label} | {u['lifted']} | {u['undetected']} | {c['lifted']} | {rel['fell_like_gt']} | {rel['vanished']} |")
lines += ['', '阈值为冻结的 %.6f 像素；按事件窗口最大上移量计算，不能称作首帧错误率。漏检是红色方块检测器未检出，不自动等同物体物理消失。' % report['rise_threshold_px'],
          '这些是小规模验证集代理指标。不能据此声称总体失败预测已解决，也不能用它代替后续独立 WM 评估。', '', '### 阈值敏感性（虚构抬升例数 /24）', '',
          '| Codec | 18px | 冻结阈值 | 25px | 30px | 40px |', '|---|---:|---:|---:|---:|---:|']
for label,r in report['codecs'].items():
    rises=[x['pred_rise_px'] for x in r['per_case'] if x['category']=='unlifted_failure']
    counts=[sum(x is not None and x>t for x in rises) for t in [18,report['rise_threshold_px'],25,30,40]]
    lines.append('| '+label+' | '+' | '.join(map(str,counts))+' |')
lines += ['', '### 未抬起失败中的残余异常', '']
for label,r in report['codecs'].items():
    xs=[x for x in r['per_case'] if x['category']=='unlifted_failure']
    abnormal=[f"{x['id']} ({x['pred_rise_px']:.2f}px)" for x in xs if x['pred_lifted']]
    lost=sum(x['vanished_frames'] for x in xs)
    visible=sum(x['gt_visible_frames'] for x in xs)
    lines.append(f"- {label}：虚构抬升案例 {', '.join(abnormal) or '无'}；GT可见窗口帧中重建未检出 {lost}/{visible}。")
lines += ['', '目视核查：mixed34k 的 g00006_b03、g00014_b06 在所看关键帧保留方块留桌状态；g00059_b03 的第57帧确实把红色方块画到夹爪下方，不只是阈值临界误判。未把少数预览视为全量人工标注。']
lines += ['', '## 最近 checkpoint 验证趋势', '',
          '以下 ROI/位置/漏检指标来自原训练验证器，事件表使用生成式解码；两套定义不同，分开报告。选模同时检查两套指标。', '',
          '| 步数 | 成功 ROI 误差 | 失败 ROI 误差 | 失败中心误差 px | 失败漏检率 | 原数值门槛 | 事件门槛 |',
          '|---|---:|---:|---:|---:|---|---|']
for c in sorted(selected['candidates'],key=lambda x:x['step']):
    s,f=c['validation']['success'],c['validation']['failure']
    lines.append(f"| {c['step']} | {s['roi_mae_255']:.3f} | {f['roi_mae_255']:.3f} | {f['center_error']:.3f} | {f['miss_rate']:.2%} | {c['numeric_pass']} | {c['event_pass']} |")
lines += ['', f"最近候选的失败 ROI 相对跨度为 {selected['stability_failure_roi_relative_spread']:.2%}，符合本次‘差不多稳定、可以开始 WM 对照’的操作标准（≤5%），不是已达到全局最优或完全收敛的证明。",
          '事件验收：未抬起组虚构抬升≤1/24、无完全漏检；成功20/20；提前释放7/7落下、0/7漏检。至少两个近期版本通过，再在通过者中按原失败综合分数选择。',
          '', '## 下一轮 WM', '',
          '- 两组：仅成功443条；混合443成功+357失败（失败比例44.6%）。',
          '- 两组都使用同一固定混合 codec；各自重新构建 token cache。',
          '- 原 RT1 初始化、seed 20261005、各10000步、global batch64、每卡batch8、原学习率及采样、每2000步保存。',
          '- 唯一实验变化是 codec 与对应缓存；未启用 E2/新 loss，也不是从旧 WM 继续训练。',
          '- 独立 bundle、缓存和输出目录；8卡机须由用户提交任务，本机没有启动8卡训练。', '',
          '```bash', 'bash /media/tyshao/fail_WM/FutureWorlds_SFT/run_mixed_codec_wm_8gpu_20261011.sh', '```', '',
          '入口依次编码成功组缓存→训练成功组→编码混合组缓存→训练混合组，支持已有合法 checkpoint 续训。', '',
          f"固定 bundle：`{selected['bundle']}`", f"权重 SHA256：`{selected['codec_sha256']}`", '',
          '配置、选择记录和 smoke 结果：`experiments_mixed_codec_wm_20261011/`。',
          '逐案例结果、关键帧及左右GT/重建MP4：`evals_codec_current_20261011/`。视频全部留在服务器。']
smoke=EXP/'smoke_result.json'
if smoke.exists():
    s=json.loads(smoke.read_text())
    lines += ['',f"CPU 数据链路 smoke：{s['state']}；真实成功/失败训练样本经新codec编码、缓存精确回读、构造SFT batch并获得有限loss；batch与逐样本loss差{s['batch_vs_individual_difference']:.3g}。未运行8卡DDP训练。"]
(OUT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT_zh.md')
