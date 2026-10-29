# 已停止：本轮方案作废

用户已纠正：需要多卡、原始8帧与随机起点。下面保留错误方案记录，禁止使用此目录继续训练。正式训练未完成第1次全局优化更新；仅独立smoke完成1步。

# RT-1 SFT → MemSPO200 实验

本轮目标：从发布 RT-1 SFT50k 初始化，进行200步MemSPO/R0训练，比较同一固定评测集上的SFT和最终RL权重。不是论文固定cohort复现。

## 入口与产物
实验根目录：/media/tyshao/Jev-WM/experiments/rt1_memspo_20261017

准备数据：scripts/prepare.py
独立功能检查：bash scripts/run.sh smoke
正式训练与自动评测：bash scripts/run.sh all
中断后继续：同一 all 命令；有完整checkpoint时自动恢复，卡数必须保持1。
状态：status.json
总日志：logs/all.log
训练逐步指标：train/metrics.jsonl
配置与协议：train.json、protocol.json、data_audit.json
结果：comparison.json、REPORT.md
固定最终权重：train/checkpoints/step_000200、rl_bundle/

## 预先固定的比较
- SFT基线：paper_assets/futureworlds_rt1 的 sft variant。
- MemSPO：200步，global batch16，单卡累积16条，horizon10，lr1e-6，R0=负的MAE+LPIPS_VGG，clip0.2，KL0.001，diversity0.1，seed20261017。
- 每10步保存恢复点；smoke的模型和优化器独立，正式训练重新从SFT开始。
- 比较固定最终step200，不使用验证或测试分数选checkpoint。
- 评测两帧观测、Beam4、FP32、TF32关闭、anchor1+recent6，不注入未来GT；报告10/20/32帧。
- 主要指标：test 32帧LPIPS。报告均值、RL−SFT、相对改善百分比、配对episode bootstrap 95%置信区间（10000次，固定种子）。
- 训练完成后自动评测SFT与RL；测试数据、超参数、checkpoint选择规则在训练前锁定。

## 数据
官方fractal20220817_data 0.1.0的前16个TFRecord分片，通过GCS MD5核验。
过滤少于34帧或T5文本超过64token的episode，按record SHA-256去重。
按固定seed+episode ID哈希排序，32条validation、64条test、随后最多512条train。
每episode仅取一个固定34帧窗口，合法起点由另一独立哈希确定。窗口开始的两帧是唯一真实初始观测。
RL读取前12帧，评测读取全部34帧。episode之间严格隔离。
本次RL测试集与原始SFT预训练数据是否重叠未知，不能将此称为全新未见数据泛化。
正式数量见data_audit.json。

## 代码和资源
复用vendor/scholar_futureworlds，逐文件核验SOURCE_LOCK和已有decoding兼容补丁。
不修改训练目标、奖励或搜索实现。
Python：/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python
当前一张A100 80GB，另有约52GB占用。本实验不停止其他进程。
CUDA smoke用于确认实际更新、推理、显存及速度；估算不包含全部评测和写盘开销。

图像预测指标不等于机器人任务成功率。单次seed、有限episode的提升或下降如实报告，不保证RL一定涨点。
