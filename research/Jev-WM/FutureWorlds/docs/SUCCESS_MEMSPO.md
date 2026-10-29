# 成功轨迹 MemSPO 训练

目的：以学长 bundle 的原始 `sft` 权重为起点，使用现有 MemSPO + R0 训练，检验是否改善成功轨迹的未来视频预测。所有输出在服务器/训练机。无需 OpenAI 密钥。

## 数据与配方

- 来源：本地 CounterCubePreview 数据，不是论文固定 128-case cohort。
- 训练：train_success.json 的全部 443 条成功轨迹。
- 验证：val.json 中 success=true 的全部 53 条轨迹。
- 测试：test.json 中 success=true 的全部 59 条轨迹。
- 不重新随机划分；验证训练 episode 和 initial_state_id 不重叠，检查源 NPZ 哈希。
- 训练取各轨迹中间连续 12 帧：2 帧初始观测 + 10 帧未来。
- 验证/测试取各轨迹中间连续 34 帧：2 帧初始观测 + 32 帧未来。
- 仅用成功轨迹并不意味着每个短窗口已经完成任务。
- 训练顺序为固定种子打乱的完整成功训练集，循环至所需样本数；验证/测试不参与排序。
- 默认 200 步，全局 batch=16，lr=1e-6，KL=0.001，clip=0.2，diversity_penalty=0.1。
- 每个初始状态 4 组、每组 2 beam，最终 4 条候选；奖励为 -mean_t(MAE+LPIPS_VGG)。
- 视觉 codec、T5 和 SFT reference 冻结；更新 world-model predictor 和文本适配器。
- AdamW；RL 实际使用统一 lr=1e-6，SFT 的 adapter_lr/warmup 不适用于本次 RL。
- 每 50 步保存 checkpoint，默认只评估最终 200 步，不按验证成绩挑 checkpoint。
- 支持 1/2/4/8/16 张卡；每卡累积 16/卡数 个样本，保持全局 batch。
- 8 卡默认每卡累积 2 个样本。单节点 torchrun DDP；不是跨节点启动脚本。
- 通过 `STEPS=400` 可改为 400 步，必须在首次准备时设定；协议固定后不能原地改步数。
- `paper_locked=false`：保留原算法/超参数，但数据和顺序不同，不宣称论文重训复现。

## CUDA 12.4 训练镜像

针对 `cuda:v2.0-cu12.4-cudnn9-ubuntu22.04`，首次在训练机执行：

```bash
cd /media/tyshao/Jev-WM/FutureWorlds
bash scripts/setup_success_memspo.sh
```

需要 Python 3.10–3.12（Ubuntu 22.04 默认 Python 3.10 可以）。若缺少 venv 模块，先安装对应 Python 的 venv 包；系统默认 Python 可用 `apt-get update && apt-get install -y python3-venv`。

脚本创建仓库内独立的 .venv-cu124，安装 PyTorch 2.5.1 / torchvision 0.20.1 的官方 cu124 wheel 和 requirements-cu124.txt，并检查依赖及每张可见 GPU 的矩阵运算。启动脚本自动优先选择此环境，无需修改数据/权重路径或手动激活环境。PYTHON 环境变量仍可显式覆盖。

镜像标签中的 v2.0 不能当作 PyTorch 版本。PyTorch wheel 自带对应 CUDA/cuDNN 运行依赖；镜像标签本身不能保证宿主机驱动及 GPU 架构兼容，应以安装后的 GPU 检查为准。不要用通用 requirements-linux.txt 覆盖此专用环境。

当前服务器只验证过 PyTorch 2.5.1+cu121 的单步训练；cu124 安装与实际多卡训练需在目标训练机验证。新环境用于从 SFT 开始本次正式实验，不续训其他运行环境的 checkpoint。

## 运行

训练机路径与当前服务器一致，脚本已默认使用现有 CUDA Python、SFT bundle、数据和 T5 路径，无需修改路径。默认 8 卡、200 步：

```bash
cd /media/tyshao/Jev-WM/FutureWorlds
bash scripts/run_success_memspo.sh all
```

如需更换卡数，可在首次运行前设置 GPUS；路径仍支持通过 PYTHON、BUNDLE、DATA_ROOT、TEXT_MODEL 环境变量覆盖。输出默认写入仓库 runs/success_memspo。

脚本顺序：筛选/校验并裁切数据 → CUDA/权重校验 → SFT 验证+测试基线 → MemSPO → 导出 RL bundle → RL 验证+测试 → REPORT.md 和 comparison.json。

仅在独立 SSH 终端中可用后台启动，日志仍保存在 OUTPUT。训练平台的启动命令必须使用上面的前台命令，不能加 nohup 或末尾的 &，否则平台可能判定任务完成并回收后台进程：

```bash
nohup bash scripts/run_success_memspo.sh all > success_memspo.launch.log 2>&1 < /dev/null &
```

分阶段运行：

```bash
bash scripts/run_success_memspo.sh prepare
bash scripts/run_success_memspo.sh preflight
bash scripts/run_success_memspo.sh all --dry-run
bash scripts/run_success_memspo.sh baseline
bash scripts/run_success_memspo.sh train
bash scripts/run_success_memspo.sh evaluate
bash scripts/run_success_memspo.sh report
```

dry-run 会准备/核对本地数据清单并打印 GPU 命令，不启动 GPU 训练或评估。数据准备会读取并核验全部成功源文件。

重复执行 all 可校验并复用已有评估、从最近完整 checkpoint 恢复训练。恢复要求相同 GPU 数、Python 依赖版本、代码、路径、数据和配置。第一次 checkpoint 前中断时脚本明确报错，需用新 OUTPUT 从 SFT 重启。不要在运行中更改代码/配置。

CUDA 12.4 镜像请使用上述专用安装脚本和 requirements-cu124.txt。首次使用 LPIPS-VGG 时需要已缓存的 VGG 权重或下载权限；离线训练机须迁移对应 PyTorch cache。

当前服务器已验证的 GPU Python 是：
`/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python`
仓库本身的 `.venv` 目前是 CPU-only，不要用于 GPU 训练。

## 4 卡运行

路径与环境选择保持一致。在训练平台的启动命令中使用前台运行，平台才能持续跟踪进程并收集标准输出：

```bash
cd /media/tyshao/Jev-WM/FutureWorlds
exec bash scripts/run_success_memspo_4gpu.sh all
```

不要在平台启动命令中加 nohup、末尾 & 或重定向全部输出；脚本本身会将输出同时写到平台控制台和 runs/success_memspo_4gpu/logs/all.log。

默认 4 卡、200 步，全局 batch=16，每卡每次更新累积 4 个样本。仍使用全部成功训练轨迹、原始 SFT 初始化、MemSPO + R0，并自动完成 SFT/RL 评估。遵循 CUDA_VISIBLE_DEVICES；未设置时使用前 4 张可见 GPU。

输出默认在 runs/success_memspo_4gpu，报告为该目录的 REPORT.md。不要将 OUTPUT 指向已有 8 卡实验目录，也不要从 8 卡 checkpoint 恢复本次 4 卡实验；可以从同一原始 SFT 重新开始。脚本支持与通用入口相同的 prepare/preflight/baseline/train/evaluate/report 阶段及 --dry-run。

## 搬到训练机

需要：完整 FutureWorlds 源码（包括新增 scripts/success_memspo.py 与 scripts/run_success_memspo.sh）、权重 bundle、T5、成功数据源 JSON 及其 cases/ 文件、LPIPS/VGG 参数、匹配的 CUDA 环境。

已提供干净源码包（不含权重、数据、环境和已有实验）：
`/media/tyshao/Jev-WM/futureworlds_success_source.tar.gz`。

源码包内容直接位于包根目录；在训练机解压到目标仓库：

```bash
mkdir -p /media/tyshao/Jev-WM/FutureWorlds
tar -xzf /media/tyshao/Jev-WM/futureworlds_success_source.tar.gz -C /media/tyshao/Jev-WM/FutureWorlds
```

原始数据目录可以包含失败 NPZ，筛选器只读取 success=true 对应文件，不把失败数据用于训练或评估。迁移时可只复制三个清单和其成功条目引用的 NPZ；不需要 cases_failure70_extra。

## 输出与判断

- protocol.json / data_audit.json：固定协议与成功样本计数。
- train.json / schedule.json：有效配置与全部训练顺序。
- train/metrics.jsonl：训练日志。
- train/checkpoints/：模型/优化器/RNG。
- evaluation/{validation,test}/{sft,rl}/：同协议预测及指标。
- REPORT.md / comparison.json：最终比较。
- 日志：logs/{阶段}.log。

主要结果：独立测试集 32 帧 LPIPS 是否低于 SFT。同时报告 MAE、PSNR、SSIM 和 10/20 帧结果。区间使用 initial_state_id 分组 bootstrap，避免把相同初始状态下的多个分支当作独立样本。小任务、单随机种子，不能据此推断真实机器人成功率。

单步 smoke 只验证运行链路，不能作为涨点证据；尚未运行正式 200 步之前，不宣称效果。
