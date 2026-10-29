# Jev-WM 短程对照实验协议

实验目录：`runs/jev_pilot_20261012`。脚本：`scripts/jev_pilot.py`。

## 固定条件

- 学长原始 RoboCasa SFT 起点、配套 codec 和 T5；FP32、MemSPO、seed=20261012。
- 20 个训练初始状态，成功/失败各 10 个；固定顺序重复两遍，共 40 个样本暴露。
- global_batch=2，20 个 optimizer steps，lr=1e-6，4 条候选，每条预测 4 帧。
- baseline 使用 R0；hybrid 使用 R0 + 0.25 × Decisions；不据验证结果调权重。
- 校准 4 个独立训练初始状态，验证 8 个独立验证初始状态，均与实际训练状态不重叠。
- 每条轨迹取中间十帧，选择规则不依赖模型分数。保留源文件 SHA、源帧起点。
- 验证三种权重：SFT、R0、hybrid；相同 Beam4、初始观测 2 帧，无 GT 刷新。
- 同时计算 4/8 帧 PSNR、SSIM、LPIPS、MAE、MSE；8 帧超过本轮训练视野。
- 固定候选校准：真实序列、乱序、单黑帧、冻结初始画面，真实序列重复评分 3 次。
- 校准揭示冻结偏好后，增加冻结初始帧的验证基线、帧间变化 MAE 和运动幅度诊断。
- 一张共享 A100，单任务 PyTorch 显存上限 25%；两组 GPU 训练顺序运行。
- 官方 API，通过回环地址 SSH 隧道连接本机 Clash，每次 HTTP 尝试至少间隔 6 秒。

## 协议修订与失败记录

最初 global_batch=1 的 R0 试跑在第 7 个样本遇到候选奖励全同，原框架终止。
在混合组启动和任何模型验证前，将两组统一改为 global_batch=2、固定序列两遍，
均从原 SFT 重启。失败输出保留为 `base_aborted_batch1`，不计入结果。
没有删除或替换该同分样本，也没有给奖励加随机扰动。

校准请求曾触发 429，成功响应已保存。增加实验级节流后续跑，缓存结果不重复收费。
节流放在实验脚本，训练实现 source fingerprint 不因该运行设置改变。

## 重现

从仓库根目录设置 `PYTHONPATH=src`、`USE_TF=0`、`USE_FLAX=0`，使用 GPU 环境：
`/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python`。
依次运行脚本阶段：`prepare`、`calibrate`、`train-base`、`train-hybrid`、`evaluate`、`analyze`。
API 阶段使用不回显的密钥输入。需要本机 Clash 和 SSH 隧道保持在线，详见 `JEV_WM.md`。
`prepare` 不覆盖已有协议；训练不覆盖已有 run，重跑须使用新实验目录或原恢复协议。

## 解释边界

这是单 seed、20 次更新、8 个验证状态的探索性实验，不能替代多 seed 或完整数据集评估。
配对 bootstrap 区间仅用于展示小样本不确定性，未做多指标校正。
合成破坏测试不能当作人工标注一致性测试；本轮没有收集人工偏好标签。
冻结序列是潜在奖励投机的探针：缺少动作语义的评审本来就无法充分判断不动是否错误。
因此若冻结分数高，说明奖励约束不足，不能单独断言评审视觉识别失败。
不以同一评审的训练分数上涨作为有效性结论。

## 官方 RPD 配额导致的范围缩减

混合组完成 2 次更新后，官方返回 `rate_limit_exceeded`：gpt-6-luna 的
requests per day 为 50，used=50，remaining=0。Retry-After 为 1728 秒，
并不意味着届时可以立即完成剩余全部请求。未换供应商或伪造分数继续训练。

原定 20 步混合组未完成，也未生成第 20 步 checkpoint。
利用已经缓存的官方评分，从原 SFT 重建双方各 2 步、global_batch=2 的对照，
混合组设 backend=replay，不再调用 API。缓存重放的前两步奖励、损失和梯度范数
与原在线运行进行逐步核对。该结果只能标为“两步匹配试验”，不能和 20 步 R0
直接比较并归因于奖励方法。20 步 R0 结果仅作为另一个训练时长的补充观察。

补充脚本：`scripts/jev_pilot_2step.py`、`scripts/analyze_jev_pilot.py`。
原始在线失败目录 `hybrid` 保留；两步产物位于 `base_2step` 和 `hybrid_2step`。
