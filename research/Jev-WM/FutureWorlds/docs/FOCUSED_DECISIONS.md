# Focused pure Decisions：兼顾图像贴合与物体行为

实验目录：runs/local_success_decisions/focused。旧诊断证据保留，未恢复旧RL权重，未修改vendor的4卡原方法实验。

## 固定方案

- 同一 local_success_sft 起点（443成功轨迹SFT10000 + success codec44000）。
- 原始首两帧，预测32帧，52维动作；FutureWorlds MemSPO4组×2beam，LR1e-6，KL0.001。
- 官方 gpt-6-luna Decisions；纯API奖励，不混入MAE/LPIPS。
- 输入为分开的参考/预测图片：8时刻完整视图 + 16时刻物体细节。裁切由参考序列确定，对所有候选相同；放大保持纵横比。
- 三个问题分别检查整体几何/运动、红方块行为、画面是否可用。
- 奖励 = sqrt(fidelity × object_behavior) × usable_scene，三个值均为归一化官方分数。物体或画面严重出错时，高背景分不能直接补偿。
- 每个候选都评参考先/预测先两个顺序，取平均；有明显顺序反转、顺序分差超过0.15或候选差小于0.04时，该组不产生策略优势。
- 概率与置信度按官方schema校验，不把置信度当错误概率；顺序一致性不等于正确性。

## 校准与对照

1. 已诊断的g00045_b09仅作开发：真实片段、黑屏、冻结、移除方块、复制方块、两个实际候选，共14次两顺序评分。人工改造的对照不会成为训练样本。
2. 随机固定3个新的训练初始状态组：各4候选+5对照，两顺序54次评分。具体ID和门槛在protocol.json，在调用前固定。与开发组和后续试训schedule按initial_state_id排除重叠。
3. 对照要求正确片段高分、黑屏低分、冻结降分，复制/消失使物体分下降；至少2组有可信分差，选中候选平均LPIPS不明显劣于候选均值（上限1.05倍）。该LPIPS门槛仅防止明显偏离图像质量，不当作行为正确率。
4. 门槛通过才运行4steps/batch2试训，保存每步；之后对最后step4和固定SFT评估同4个validation组，horizon10/20/32，不选最优checkpoint。
5. 还需人工核对SFT/RL的视频或逐帧局部图，报告复制/消失/运动时间错误，不能只凭奖励上升宣称提升。

所有API请求限量，间隔6秒，出错停止，不伪造分数。develop上限16、calibrate56、train72（已包含双顺序）。代理沿用服务器17897到本机Clash7897，密钥只在私有配置。

运行：
```bash
cd /media/tyshao/Jev-WM/FutureWorlds
export USE_TF=0 USE_FLAX=0 PYTHONPATH=src
/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python scripts/focused_decisions.py all
```

可以分阶段使用 dev_images / develop / collect / calibrate / train / evaluate。train --resume从完整checkpoint恢复。每个阶段日志在focused/logs，API包与回执在focused/reviews。

当前新增实现为src/futureworlds/focused_alignment.py，接入已有训练分支；旧reference_alignment保留供原实验复核。局部裁切针对当前红方块RoboCasa，不可直接泛化到其它任务。

限制：小样本开发验证、无盲测结论；16个局部时刻仍可能漏瞬时错误，双顺序一致也可能稳定地犯错。没有承诺指标和物体行为同时提高。


本轮补充：初次3组控制测试全通过；旧的顺序判定把低分候选间0.046/-0.008的小幅反向差异当成整组失败。现改为：反向差异在两种顺序下都>=0.04才否决；否则把这两个候选归成同分。仍保持order_delta<=0.15和最终range>=0.04的门槛。
初次结果及协议保存在initial_calibration.json / initial_protocol.json；新固定3个训练初始状态g00023_b00、g00020_b09、g00053_b01，均避开此前开发/校准组及试训schedule，作为修改规则后的重新检查。不会把对旧样本的重算称为独立验证。

