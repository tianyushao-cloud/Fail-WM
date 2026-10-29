# 纯 Decisions RL 与原始 SFT 对比

直接使用学长原始 SFT 权重作为起点和基线，没有重新训练 SFT。

**当前未证明优于 SFT。** 一步更新后，4/8 帧平均 LPIPS 分别变差约 0.32%/0.92%，但两者差值区间均跨过 0；8 帧 MAE 略有改善，其他指标涨跌不一。只有 1 步、8 个样本，不能据此判断纯 Decisions 长期是否有效。

训练奖励仅为 OpenAI Decisions 的物体一致性与时序交互评分均值；没有 R0/LPIPS/MAE 奖励。保留原 MemSPO 的 KL 正则。目标 20 步，实际完成 **1 步**；每步保存 checkpoint。

固定 8 个独立验证初始状态，Beam=4，预测 4/8 帧，不注入未来真实帧。SFT 评估复用先前结果，已核对协议、模型资产和预测文件哈希。

| 预测长度 | 指标 | 原始 SFT | 纯 Decisions RL | RL−SFT | RL 更好样本数 |
|---|---|---:|---:|---:|---:|
| 4 帧 | lpips ↓ | 0.218889 | 0.219583 | +0.000694 | 5/8 |
| 4 帧 | mae ↓ | 0.045509 | 0.045601 | +0.000093 | 5/8 |
| 4 帧 | psnr ↑ | 19.962977 | 19.966635 | +0.003657 | 5/8 |
| 4 帧 | ssim ↑ | 0.814078 | 0.813340 | -0.000738 | 3/8 |
| 8 帧 | lpips ↓ | 0.239290 | 0.241497 | +0.002206 | 5/8 |
| 8 帧 | mae ↓ | 0.050766 | 0.050517 | -0.000250 | 6/8 |
| 8 帧 | psnr ↑ | 19.315023 | 19.313735 | -0.001289 | 5/8 |
| 8 帧 | ssim ↑ | 0.797872 | 0.796249 | -0.001624 | 4/8 |

LPIPS 差值的配对 bootstrap 95% 区间（单种子、8 样本；未经多重比较校正）：
- 4 帧：[-0.0002832619356922805, 0.002057752362452446]
- 8 帧：[-0.0008897042949683963, 0.007221617602044713]

| 8 帧运动诊断 | SFT | 纯 Decisions RL |
|---|---:|---:|
| motion_roi_mae | 0.216347 | 0.216505 |
| predicted_motion | 0.007879 | 0.007876 |
| true_motion | 0.013643 | 0.013643 |
| delta_mae | 0.018867 | 0.018891 |

运动区域按 GT 相对初始帧的平均 RGB 差 >0.02 定义，仅用于验证诊断。运动幅度不是越大越好。

当前评分规则此前在 4 组静止负例中有 3 组偏好冻结帧；此轮按要求保留纯 Decisions 奖励进行检验。全图指标易受静态背景影响，不能替代物体运动/接触或机器人成功率。没有人工偏好标注。

API 新成功调用：1；新输入 token：943。已有官方评分仅在完整输入内容哈希一致时复用。

训练状态：STOPPED。Official Decisions HTTP 429; no score produced

实际 API 限流证据：`[{"status": 429, "code": "rate_limit_exceeded", "headers": {"retry-after": "1728", "x-ratelimit-limit-requests": "50", "x-ratelimit-limit-tokens": "100000", "x-ratelimit-remaining-requests": "0", "x-ratelimit-remaining-tokens": "39607", "x-ratelimit-reset-requests": "23h45m35.273s", "x-ratelimit-reset-tokens": "434h49m46.56s"}}]`

Checkpoint：`/media/tyshao/Jev-WM/FutureWorlds/runs/jev_pure_20261012/train/checkpoints/step_000001`

仅为小规模探索。是否优于 SFT 以验证结果为准，不代表长期训练效果或任务成功率改善。

验证：9 项 Decisions 相关测试通过；checkpoint 和评估预测哈希通过；111/111 核心参数张量及 24/24 文本适配器张量有实际变化，均为有限值。未重跑无关完整测试套件，先前记录的间歇性 CPU SFT 测试问题仍未定位。

恢复：配额恢复或提高后，重新建立代理隧道，使用原环境执行 `PYTHONPATH=src USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false python scripts/jev_pure.py train --resume`，从第 1 步 checkpoint 的优化器和 RNG 状态继续。API key 通过环境变量或隐藏输入提供。

![LPIPS 对比](comparison.png)

![固定第一个验证样本，非挑选](qualitative_00.jpg)
