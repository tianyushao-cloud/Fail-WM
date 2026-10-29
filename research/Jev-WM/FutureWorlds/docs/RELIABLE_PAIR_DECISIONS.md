# Reliable-pair Decisions 实验

本方案是对 pure Decisions 训练信号筛选的修改，不声称复现论文原始奖励。
基座仍是 local_success_sft（443-success SFT step10000 + success codec step44000），原始首两帧条件、32帧预测、成功轨迹、MemSPO 候选生成和原有 KL。

## 为什么只学习一对

旧流程将四个候选整体排序，只要其中一个候选受呈现顺序影响就否决整组。
新流程复用 focused_alignment 的官方 Decisions 三项分数，在两种参考/预测顺序中寻找同一可靠候选对：

- 两个顺序下，综合分差均 >=0.04，物体行为分差均 >=0.04。
- 胜者的几何分和画面可用性分在两个顺序下均不低于败者。
- 仅检查入选两条候选：每个评分分量的顺序差均 <=0.15。
- 胜者在两顺序的物体分均 >=0.50，避免从全是严重错误的轨迹里硬挑“最好”。
- 有多个合格对时，取两顺序中最小综合分差最大的对；最优对精确并列则弃权。
- 赋值：胜者 +1，败者 -1，其他候选 0。组均值严格为0，未入选者的 GRPO 偏好 advantage 为0，原有 KL 仍保留。
- 没有可靠对时全0；一个全局 batch 完全无信号仍停止，不靠放宽门槛或虚构标签制造更新。

所有奖励判断来自 Decisions。LPIPS/MAE/颜色连通域均不作为训练奖励；LPIPS仅作离线诊断与小试训前的图像退化筛查。
呈现顺序一致不等于判断正确，因此另做未用于本次开发的初始场景校准和可视检查。

## 文件与运行

- src/futureworlds/reliable_pair.py：候选对筛选、训练奖励及原始回执关联。
- scripts/reliable_pair_decisions.py：协议、候选采集、API校准、训练和固定SFT/RL比较。
- tests/test_reliable_pair.py：包含真实 GRPO advantage 的零梯度检查、顺序/候选置换、错误排序/全坏/全同分拒绝。
- runs/local_success_decisions/reliable_pair/protocol.json：开始采集前固定的3个校准初始场景和门槛。
- runs/local_success_decisions/reliable_pair/calibration.json：实际是否通过；不通过不能启动 train。
- runs/local_success_decisions/reliable_pair/audit.log：采集和API评分日志。
- runs/local_success_decisions/reliable_pair/development_replay.json：旧6组回放，仅开发用途。
- runs/local_success_decisions/symmetric_probe：两图对称单帧比较，21次请求；仍有顺序偏差。
- runs/local_success_decisions/observable_probe：单图可见事实识别，24次请求；细小物体对放大尺寸敏感，未接入奖励。

运行环境：
    cd /media/tyshao/Jev-WM/FutureWorlds
    USE_TF=0 USE_FLAX=0 PYTHONPATH=src:scripts \
      /media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python -u \
      scripts/reliable_pair_decisions.py audit_run

train 阶段要求校准通过且已有与协议/校准文件校验和绑定的可视检查记录。
脚本自动从服务器私有文件加载密钥，不在命令行、代码或报告中写密钥。
原4卡R0实验和 vendor/scholar_futureworlds 不受本方案修改。

