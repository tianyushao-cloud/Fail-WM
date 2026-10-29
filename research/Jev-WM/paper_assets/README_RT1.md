# RT-1 论文资产核查与下载记录

检查日期：2026-10-17（Asia/Shanghai）。

## 已就绪的模型资产
- 发布仓库：https://huggingface.co/easylearning/FutureWorlds
- 固定 revision：64b9802cb328b66c8ea263f2a4acd795c28bb144
- 本地 bundle：/media/tyshao/Jev-WM/paper_assets/futureworlds_rt1
- RT-1 发布文件：26 个，3,107,336,677 字节。
- 五个预测器版本：sft（50,000 步）、memspo200、grpo200、ordinary200、main（MemSPO 500 步）。
- 配套内容：每个版本的训练后文本适配器、视觉 codec、动作范围、配置、模型卡与训练配方。
- 所有文件均通过发布 manifest 的大小和 SHA-256 校验；12 个 release_file 哈希也与现有代码仓库 weights_manifest.json 一致。
- 上游完整 manifest 保存在 audit_rt1/upstream_manifest.json；bundle/manifest.json 为保留原始文件哈希的 RT-1 子集清单，可直接被 verify_bundle 接受。
- 五个模型均通过 CPU strict state_dict 加载；codec、[13,2] 动作范围和 T5 文本编码成功加载。没有启动 GPU 训练或论文评测。

## 复用的外部依赖
- T5：/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base
- T5 revision：a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1；6 个本地文件均与固定 revision 的 Git/LFS 哈希一致。
- VGG：/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth；SHA-256 已核验匹配 torchvision 文件名校验前缀。
- Python：/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python，torch 2.5.1+cu121。
- 原版代码入口：/media/tyshao/Jev-WM/FutureWorlds/vendor/scholar_futureworlds/src
- 可加载模型的路径配置见 asset_paths.json。

## 数据仍需补齐的部分
检查了作者公开 GitHub main 树、releases、issues、Hugging Face 模型文件和作者 datasets 列表。未发现论文原始训练/评测窗口清单或对应 NPZ 发布入口。模型卡明确不包含原始数据。GitHub 中仅提供 128 个固定评测窗口 ID 和 8,000 个有序训练窗口 ID；训练清单的 8,000 个窗口和 episode 前缀都不同。

已核查官方 RT-1 数据：
- gs://gresearch/robotics/fractal20220817_data/0.1.0/
- 1,024 个 TFRecord 分片，连同元信息共 119,259,986,287 字节。
- 完整对象索引、dataset_info.json、features.json 已保存到 audit_rt1/。
- 下载并通过官方 MD5 校验了第 0 个分片（106,467,095 字节），保存在 rt1_data_probe/。
- 解析其中 80 个完整 episode，其图像分辨率为 320×256；以原始 record SHA-256 作为论文 episode ID 的假设未找到匹配。
- 这只说明此次分片探查不能确定映射，不证明所有官方分片都无法匹配。论文没有公开足以确认 ID 构造方法的清单；不能把探查分片或重新随机抽样的数据称为论文固定 cohort。
- 未下载剩余约 119GB 原始数据，未创建伪造的论文 train/evaluation manifest。

严格复现需要的最小补充：
1. RT-1 train_windows.jsonl、dev_windows.jsonl（或作者导出的无损 NPZ + manifest）；
2. ID、split、path、TFRecord offset/length、record sha256、start、window_frames；
3. 官方源数据版本/分片映射，以及 episode/window ID 构造规则；
4. 128-case 评测清单和 8,000 次后训练 schedule 能逐项命中；
5. 若重新训练 SFT50k，还需要完整原始 SFT 训练划分及窗口构造规则。

## 校验记录
- audit_rt1/weights_verification.json
- audit_rt1/repository_hash_crosscheck.json
- audit_rt1/external_dependencies_verification.json
- audit_rt1/loading_verification.json
- audit_rt1/official_data_probe.json
- audit_rt1/rt1_official_object_inventory.json

下载脚本 download_rt1.py 支持断点续传和完整文件复用，使用服务器已有香港代理。既有 FutureWorlds 工作区、训练结果与本地适配 SFT 均未改动。
