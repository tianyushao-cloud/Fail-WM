# Fail-WM

让动作条件世界模型忠实预测成功和失败轨迹，减少“无论动作如何都预测成功”的偏差。

- **代码、实验配置和结果：** [私有 GitHub 仓库](https://github.com/tianyushao-cloud/Fail-WM)
- **精选权重：** [私有 Hugging Face 模型仓库](https://huggingface.co/tianyushao1124/Fail-WM)
- 快照日期：2026-10-29。代码、配置和历史报告按原样保存；迁移不改写实验结果。

## 保留内容

| 材料 | 位置 | 说明 |
|---|---|---|
| 最新训练库 | `src/` | 来自纯轨迹奖励 RL30 的实际源码快照 |
| 各版本代码、启动脚本、配置 | `research/` | 保持原目录层次；历史实验使用各自的 `source/src` |
| 结果和少量视频样例 | `reports/`、`research/` | 包括指标、逐样本结果、裁判记录、奖励校准和坏例 |
| 权重清单和环境 | `migration/` | SHA256、原路径、实验索引、实际训练依赖版本 |
| 下载、校验、恢复 | `tools/fetch_weights.py`、`tools/restore_project.py` | 恢复工具不访问原服务器 |
| 原上游 Git 历史 | `migration/upstream-history.bundle` | 未提交的工作区修改另存于 `research/` |

HF 精选权重约 **20.2 GiB**，包括：

- 原始 RoboCasa／RT-1 发布 bundle，以及实际使用的 RT-1 初始化。
- mixed codec34k 及其完整训练状态。
- 混合数据 SFT10k、成功-only SFT10k 的 bundle 和完整 checkpoint。
- V0 状态奖励 RL30、V6 状态奖励 RL30、纯轨迹奖励 RL30 的完整 checkpoint。
- 离线推理需要的固定版本 T5。

**不上传完整原始数据、所有历史中间 checkpoint、大批候选视频或重复的 Qwen 权重。**
数据 split/manifest 和采集代码已保留；再次训练、重算视频指标仍需自行准备数据。
`migration/full_inventory_*` 记录最初的全面盘点，不代表这些文件都上传了。
**实际上传范围以 `migration/weights_manifest.json` 为准。**

## 最新模型与已有证据

| 模型 | 训练 | 正式评测状态 |
|---|---|---|
| mixed SFT10k + codec34k | 已完成 | 作为主基线，已有 val/test |
| V0 state RL30 | 已完成 | 已有 sibling val/test；乐观率与图像质量须分开看 |
| V6 state RL30 | 已完成 | 本快照未找到正式 val/test 预测和效果评测 |
| trajectory RL30 | 已完成 | 本快照未找到正式 val/test 预测和效果评测 |

不能把离线奖励排序改善当作 RL 模型已经改善。详细勘误在
[`migration/KNOWN_ISSUES.txt`](migration/KNOWN_ISSUES.txt)，包括旧 `resume_from`
未生效、确定性 beam 的重复 seed、窗口 lift 与最终 success 的区别等。

轨迹奖励：每帧 cube mask 偏差、GT 夹爪 patch 模板匹配位移、cube–gripper 相对位置偏差，
权重 `.5/.25/.25`；无 VLM。它仍是固定相机下的二维启发式，存在遮挡与匹配误差。

## 新机器使用

```bash
# 需要对两个私有仓库有读取权限；不要把 token 写进源码。
git clone https://github.com/tianyushao-cloud/Fail-WM.git
cd Fail-WM
python -m pip install huggingface_hub
hf auth login
python tools/fetch_weights.py --output ./weights
python tools/restore_project.py --weights ./weights --destination /data/fail-wm-restored
```

恢复工具逐文件检查 SHA256，并拒绝覆盖内容不同的文件。它将代码、配置和所选权重恢复到
原相对目录；未选中的数据不会凭空恢复。

原实验使用绝对根路径 `/media/tyshao`，且路径参与部分续训指纹。
推荐在新容器里挂载 `-v /data/fail-wm-restored:/media/tyshao`，保持历史配置不变。
新实验使用新的 output 目录，不删除旧完成标记来覆盖历史。

按 `migration/training_environment.json` 和 `requirements-reproduction.txt` 安装环境。
实际 RL 环境使用 torch2.5.1+cu121、torchvision0.20.1+cu121、transformers4.57.6；
原快照中旧 `requirements-linux.txt` 不代表实际 GPU 训练环境。
历史运行默认 Python 路径是 `fail_WM/FutureWorlds_workspace/code/.venv/bin/python`；
在新机器重建该环境，或使用脚本支持的 `PYTHON` 覆盖。

## 导出最新轨迹 RL 的推理 bundle

安装项目依赖后，从本仓库执行以下 CPU 文件导出；推理时再使用 GPU：

```python
from futureworlds.exporting import export_model

export_model(
    "weights/bundles/robocasa_mixed_sft10k",
    "robocasa",
    "weights/checkpoints/robocasa_rl_trajectory_step000030",
    "trajectory30_bundle",
)
```

用 `PYTHONPATH=src` 导入本地库，或执行 `pip install -e '.[full]'`。
`WorldPipeline` 的模型变体用 `main`，文本模型指向 `weights/dependencies/t5-base`。
推理输入仍需自己提供：前2帧 RGB、对齐的52D动作和文本。

## 数据与动作定义

- 场景为自定义 `CounterCubePreview`，不是完整 RoboCasa benchmark。
- batch1000 按100个初态组划分：train/val/test = 800/100/100。
- 原始记录20Hz、640×480；模型NPZ为5Hz、320×256。
- 实际模型输入是4个原生12D动作各补1个0，串接为52D，**不是** `target13`。
- `actions[t]` 控制 frame t → t+1；native gripper `+1=close, -1=open`。
- 轨迹奖励使用 HDF5 的物体／夹爪位姿和固定相机投影，索引 `(k+2)*pack`。
  这些 GT 信息只进入奖励，不进入预测生成。

公共裁判模型 Qwen3-VL-8B 的固定 revision 见 `migration/public_dependencies.json`。
纯轨迹奖励和世界模型推理不需要加载 Qwen。OpenAI 路线需要另行配置 API 凭证，凭证不归档。
