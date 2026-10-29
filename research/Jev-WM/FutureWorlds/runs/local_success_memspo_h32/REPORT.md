# 成功轨迹 MemSPO 与本地适配 SFT 对照

本实验使用本地 CounterCubePreview 成功轨迹，不是论文固定 128-case 复现。训练与验证/测试按初始状态分离。
训练：RT-1 起点、成功数据 SFT 10000步权重初始化，MemSPO + R0，200 步，全局 batch=16，horizon=32，lr=1e-6，KL=0.001。R0 为负的 MAE 与 LPIPS 之和，不调用 Decisions。
评估：按验证集32帧LPIPS选择第100步，Beam4，FP32，两个初始观测，不注入未来真实帧。主要指标为按初态划分的测试集 32 帧 LPIPS；同时报告 10/20/32 帧各指标。

| 划分 | 帧数 | 指标 | SFT | RL | RL−SFT |
|---|---:|---|---:|---:|---:|
| validation | 10 | lpips | 0.022525 | 0.023425 | +0.000900 |
| validation | 10 | mae | 0.009317 | 0.009321 | +0.000005 |
| validation | 10 | psnr | 30.193967 | 30.198420 | +0.004453 |
| validation | 10 | ssim | 0.955957 | 0.956094 | +0.000137 |
| validation | 20 | lpips | 0.027288 | 0.028034 | +0.000746 |
| validation | 20 | mae | 0.010376 | 0.010313 | -0.000063 |
| validation | 20 | psnr | 28.553345 | 28.776850 | +0.223505 |
| validation | 20 | ssim | 0.948237 | 0.949298 | +0.001062 |
| validation | 32 | lpips | 0.036863 | 0.034776 | -0.002087 |
| validation | 32 | mae | 0.012798 | 0.012170 | -0.000628 |
| validation | 32 | psnr | 27.001065 | 27.372217 | +0.371152 |
| validation | 32 | ssim | 0.934202 | 0.937514 | +0.003311 |
| test | 10 | lpips | 0.022886 | 0.022066 | -0.000820 |
| test | 10 | mae | 0.009191 | 0.009131 | -0.000060 |
| test | 10 | psnr | 30.625171 | 30.610102 | -0.015069 |
| test | 10 | ssim | 0.957256 | 0.958367 | +0.001111 |
| test | 20 | lpips | 0.028057 | 0.027019 | -0.001038 |
| test | 20 | mae | 0.010428 | 0.010208 | -0.000221 |
| test | 20 | psnr | 28.801003 | 28.850500 | +0.049497 |
| test | 20 | ssim | 0.948687 | 0.949975 | +0.001288 |
| test | 32 | lpips | 0.039132 | 0.037603 | -0.001529 |
| test | 32 | mae | 0.013349 | 0.013046 | -0.000303 |
| test | 32 | psnr | 27.022413 | 27.136662 | +0.114250 |
| test | 32 | ssim | 0.933080 | 0.934836 | +0.001756 |

测试集 32 帧 LPIPS 均值改善；RL−SFT=-0.001529，按初始状态分组 bootstrap 95% 区间=[-0.0037415493910976897, 0.0010192173334891046]。
单种子、小规模任务；其余指标为描述性结果。区间跨零时不能宣称稳健提升。
Checkpoint按验证集32帧LPIPS选择；测试集只评估所选权重。RL 的预测指标改善不等于机器人任务成功率提升。

Base: local success-only SFT step10000; paired success-only codec step44000. SFT and RL reuse the same 443 training episodes. No train episode or initial-state overlap with the 53/59 success validation/test episodes.
Released unmodified MemSPO/R0 trainer; 200 updates; configured rollout horizon=32; Every trajectory uses its original first two observations (source_start=0); fixed upstream episode sampler. RL consumes first 34 frames (32 generated), evaluation first34 (32 generated), no GT refresh.
Validation used to choose codec, and validation/test have been inspected in earlier experiments; this comparison is not new blind generalization evidence.


## 成功标签语义补充（原始元数据核查）

success=true是整条轨迹最终结果，不能保证原始前32帧窗口内没有失败尝试。训练443条含73条recovery轨迹，其中55条记录grasp_attempt_failed；验证53条中6条、测试59条中7条也有该事件。g00052_b09首次失败4.85秒，成功13.8秒，当前预测仅覆盖0.4–6.6秒。故此前“纯成功数据”须理解为“最终成功轨迹”，不是“仅顺利成功动作”。这不推翻已有图像指标，但改变了对训练数据和行为改善的解释。详见sample_review/success_semantics_audit.json（相对实验根目录）。
