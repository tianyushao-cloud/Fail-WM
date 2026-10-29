# 成功轨迹 MemSPO 与本地适配 SFT 对照

本实验使用本地 CounterCubePreview 成功轨迹，不是论文固定 128-case 复现。训练与验证/测试按初始状态分离。
训练：RT-1 起点、成功数据 SFT 10000步权重初始化，MemSPO + R0，200 步，全局 batch=16，horizon=10，lr=1e-6，KL=0.001。R0 为负的 MAE 与 LPIPS 之和，不调用 Decisions。
评估：固定最后一步，Beam4，FP32，两个初始观测，不注入未来真实帧。主要指标为按初态划分的测试集 32 帧 LPIPS；同时报告 10/20/32 帧各指标。

| 划分 | 帧数 | 指标 | SFT | RL | RL−SFT |
|---|---:|---|---:|---:|---:|
| validation | 10 | lpips | 0.022525 | 0.023990 | +0.001464 |
| validation | 10 | mae | 0.009317 | 0.009534 | +0.000217 |
| validation | 10 | psnr | 30.193967 | 30.072057 | -0.121909 |
| validation | 10 | ssim | 0.955957 | 0.954622 | -0.001335 |
| validation | 20 | lpips | 0.027288 | 0.028693 | +0.001405 |
| validation | 20 | mae | 0.010376 | 0.010612 | +0.000236 |
| validation | 20 | psnr | 28.553345 | 28.430932 | -0.122412 |
| validation | 20 | ssim | 0.948237 | 0.947081 | -0.001156 |
| validation | 32 | lpips | 0.036863 | 0.037416 | +0.000553 |
| validation | 32 | mae | 0.012798 | 0.012943 | +0.000146 |
| validation | 32 | psnr | 27.001065 | 26.805630 | -0.195435 |
| validation | 32 | ssim | 0.934202 | 0.933125 | -0.001077 |
| test | 10 | lpips | 0.022886 | 0.021424 | -0.001461 |
| test | 10 | mae | 0.009191 | 0.008991 | -0.000200 |
| test | 10 | psnr | 30.625171 | 30.429755 | -0.195416 |
| test | 10 | ssim | 0.957256 | 0.957533 | +0.000277 |
| test | 20 | lpips | 0.028057 | 0.026809 | -0.001248 |
| test | 20 | mae | 0.010428 | 0.010216 | -0.000212 |
| test | 20 | psnr | 28.801003 | 28.624015 | -0.176988 |
| test | 20 | ssim | 0.948687 | 0.948534 | -0.000154 |
| test | 32 | lpips | 0.039132 | 0.036903 | -0.002229 |
| test | 32 | mae | 0.013349 | 0.012892 | -0.000456 |
| test | 32 | psnr | 27.022413 | 27.026844 | +0.004431 |
| test | 32 | ssim | 0.933080 | 0.934229 | +0.001149 |

测试集 32 帧 LPIPS 均值改善；RL−SFT=-0.002229，按初始状态分组 bootstrap 95% 区间=[-0.005394454666710207, 0.0003959434862732148]。
单种子、小规模任务；其余指标为描述性结果。区间跨零时不能宣称稳健提升。
没有按验证分数挑选 checkpoint。RL 的预测指标改善不等于机器人任务成功率提升。

Base: local success-only SFT step10000; paired success-only codec step44000. SFT and RL reuse the same 443 training episodes. No train episode or initial-state overlap with the 53/59 success validation/test episodes.
Released unmodified MemSPO/R0 trainer and 200-step matched-ablation recipe; Every trajectory uses its original first two observations (source_start=0); fixed upstream episode sampler. RL consumes first 12 frames (10 generated), evaluation first34 (32 generated), no GT refresh.
Validation used to choose codec, and validation/test have been inspected in earlier experiments; this comparison is not new blind generalization evidence.
