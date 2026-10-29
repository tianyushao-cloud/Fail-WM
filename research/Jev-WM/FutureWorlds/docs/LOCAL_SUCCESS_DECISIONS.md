# Local success SFT + pure OpenAI Decisions

目录：`runs/local_success_decisions`。与 `runs/local_success_memspo_h32` 分开；不改 vendor 学长训练代码。

- 起点：443 条成功轨迹 SFT step10000，配套 success codec step44000。无需重做 SFT。
- 条件：原始 episode 的第 0/1 帧，预测 32 帧（2..33），52 维动作保持不变。
- 算法：现有 FutureWorlds Decisions 分支的 MemSPO，4 groups × 2 beams，返回 4 条候选；LR1e-6、KL0.001、clip0.2。
- 奖励：官方 gpt-6-luna Decisions 对机器人位置、物体状态、场景完整性三个问题的归一化分数均值。不掺 MAE/LPIPS。
- 真实未来只交给评分器；策略生成不使用未来图像。评分器看 8 个固定同步时刻的左右参考/预测图，共 8 张图。接口没有原生视频输入。
- 训练组分差低于 0.03 时归为同分，防止把很小的打分噪声标准化成大优势。整个 batch 同分则停止，不伪造奖励。
- 校准：随机种子 20261015 预先固定 4 个不同训练初始状态组，每组 4 候选 + 真实/冻结/黑屏。4 组从后续 8 条试训 schedule 中排除。门槛保存在 protocol.json，先固定后调用 API。
- 只有校准通过才试训 4 updates，batch2，逐步存盘。校准调用上限32，训练上限40（含重试；当前禁自动重试），间隔6秒。HTTP错误停止；Retry-After 不会被忽略。
- 对比：固定4个 validation初始状态组，beam4，horizon10/20/32，最后 checkpoint 对照同一 SFT，不选最好 checkpoint，不宣称盲测或统计显著提升。
- 目前 API 奖励分支限制单进程。服务器显存上限25%；正式4卡 MemSPO任务可以独立运行。

运行（所有文件在服务器）：

```bash
cd /media/tyshao/Jev-WM/FutureWorlds
export USE_TF=0 USE_FLAX=0 PYTHONPATH=src
PY=/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
$PY scripts/local_success_decisions.py all
```

也可以分别用 prepare / collect / calibrate / train / evaluate；train --resume 恢复最后完整 checkpoint，同配置、同实现指纹，不能跨配置恢复。重复 all 不会自动覆盖已有训练，已完成的候选有校验和。

密钥从 OPENAI_API_KEY 或服务器私有 /root/.config/jev-wm/openai.key 读取，不进代码、日志或报告。默认服务器代理127.0.0.1:17897，需要本机 Clash7897 的 SSH 转发持续在线：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -p 8081 \
  -R 127.0.0.1:17897:127.0.0.1:7897 root@10.20.0.71
```

报告：runs/local_success_decisions/REPORT.md；原始图像和评分凭证在 calibration/ 与 reviews/。
早期实验依据：docs/JEV_PURE_20STEP_RESULTS_20261012.md、runs/jev_decisions_diagnosis_20261013/REPORT.md。旧实验的短窗口和学长 SFT 不作为新实验初始化。

限制：校准样本很少；同组候选对互相关联。用 LPIPS 核对奖励排序只作离线诊断，不进入训练奖励。黑屏控制不能代替目标物体消失控制，固定8时刻也可能漏掉中间帧错误。未通过校准时不应长训，更不能保证 RL 会优于 SFT。

