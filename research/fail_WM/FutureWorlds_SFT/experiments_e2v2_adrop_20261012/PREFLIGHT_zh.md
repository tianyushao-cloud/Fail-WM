# E2v2 × action dropout 前置核验

回复「[2026-10-12 11:10] Opus — 执行指令：P3 E2 v2 + P6 动作 dropout」。用户本轮要求查看新任务，已进行CPU前置可行性检查；未启动GPU任务、未训练B/C/D，未改原训练源码/cache/冻结评估。

**事实：原方案没有通过自己规定的开训门槛。** 640000个cursor逐个调用原`Schedule.pick`，保持轨迹ID不变，按指定事件窗口 + lift/early_release阶段窗口并集，以50%概率替换target。关键202条失败的`[L,L+6]`命中18991→47483，**2.5003倍，低于3倍**。依照执行指令“达不到就先别训，在这里说明”，本轮保持HOLD_COVERAGE_GATE，不将提案schedule发布为可开训配置。

| 分组 | 轨迹数 | 样本数 | 原均匀命中次数（占比） | E2v2命中次数（占比） | 倍数 |
|---|---:|---:|---:|---:|---:|
| success | 443 | 354628 | 43381（12.23%） | 79388（22.39%） | 1.8300 |
| failure | 357 | 285372 | 34681（12.15%） | 79131（27.73%） | 2.2817 |
| failure_type/grasp_miss | 139 | 111276 | 15319（13.77%） | 40528（36.42%） | 2.6456 |
| failure_type/contact_without_lift | 93 | 74113 | 7573（10.22%） | 15682（21.16%） | 2.0708 |
| failure_type/dropped_before_goal | 45 | 36093 | 3885（10.76%） | 7578（21.00%） | 1.9506 |
| failure_type/lifted_but_not_placed | 60 | 47894 | 6563（13.70%） | 12057（25.17%） | 1.8371 |
| failure_type/object_off_counter | 20 | 15996 | 1341（8.38%） | 3286（20.54%） | 2.4504 |
| gripper/0 | 48 | 38420 | 5067（13.19%） | 11638（30.29%） | 2.2968 |
| gripper/+1 | 752 | 601580 | 72995（12.13%） | 146881（24.42%） | 2.0122 |
| critical_closed_never_lift_failure | 202 | 161402 | 18991（11.77%） | 47483（29.42%） | 2.5003 |

表中占比的分母为该组被选中的全部训练样本，不是窗口大小。800条均有attempt0/lift，缺失lift阶段为0；early_release只在存在时添加。关键组定义为train失败、整条event_summary.lift为false、原始动作索引对齐的lift阶段每个gripper值均为+1；正好202条。phase gripper从打包动作reshape(-1,13)后按原始first_action:last_action+1切片，避免跨阶段边界。此定义下+1且曾抬起为550条（Opus此前写495），这项分组口径需要对齐；关键202条门槛不受影响。

**原因与建议（不是已采用配置）**：多个事件的并集仍包含很多关键抬升窗口之外的帧。对202条关键失败，完全从并集取帧时，按实际episode抽样次数计算的期望命中为76074.50；保留50%原均匀部分只能达到约2.5倍。只调整替换概率时，期望达到3倍需p约0.6654。建议Opus将p=0.70作为新候选，保持成功/失败统一规则，再跑一次实际64万cursor门槛核验；或重新分配lift/early_release/其他事件的采样权重。**本次未擅自修改50%协议，也未把期望当成实际通过。**

**动作dropout可行性**：全量读取现有800条cache，context移位后范围4375..8749、dynamic为0..4374、action为8750..9005，9006出现次数0，符合借用BOS作null的前置条件。这仅证明不与现有训练token冲突，尚未证明训练收益或通过GPU smoke。独立cursor随机规则的计划dropout比例：总体25.0491%、成功25.0508%、失败25.0470%；这些是CPU计划掩码计数，不是已经训练的日志。

**兼容性发现**：`protocol.implementation()`递归hash整个futureworlds目录内的所有.py；即使仅新增batched_sft_adrop.py也会改变feature fingerprint，使原cache不匹配。后续动作dropout入口应放在SFT根目录，单独把新入口/模块SHA写入新实验fingerprint，保持原源码和旧行为不变。

**产物**：`preflight_e2v2_adrop_20261012.py`、`experiments_e2v2_adrop_20261012/preflight.json`、`episode_windows.json`、`schedule_e2v2_proposed.json`（仅提案，未达门槛）、`PREFLIGHT_zh.md`。schedule RNG为`random.Random(seed+cursor*104729+20261012)`；dropout计划为`random.Random(seed+cursor*130363+60261012)`。原seed20261005。新训练入口、GPU smoke、8卡启动脚本和训练本轮均未完成；待采样协议修订后继续。当前没有专用8卡队列的直接提交接口，后续仍需准备一次提交入口交用户提交，不承诺已排上8卡。

**证据边界保持不变**：支持研究夹爪指令捷径，但166/180统计的是多帧窗口最大抬升，不能直接写成“下一帧错误率”或证明完全不看画面；与新codec的start/+2对照应使用Opus正在生成的A结果，不能将旧codec的+2/+4合计当直接基线。动作dropout也可能降低动作敏感性，需以冻结的成功/失败护栏评估，不能预设有效。

第3节本任务状态更新为“CPU前置检查完成，关键曝光2.5003倍未过3倍门槛；未开训”。其他任务及对方条目未改。
