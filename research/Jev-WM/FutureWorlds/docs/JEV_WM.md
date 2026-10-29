# Jev-WM：Decisions 奖励实验

已实现官方 Decisions 多图评分、严格评分回放、候选导出与混合奖励。
协议依据：[OpenAI Decisions 文档](https://developers.openai.com/api/docs/guides/decisions)。
官方支持的是文本和图片；本实现将全部初始帧和预测帧按时间顺序作为多图输入，
不宣称支持原生视频上传。只有初始观测和预测帧会发给评审，未来 GT 留在原奖励内部。

## 配置

`configs/train/bridge_jev_hybrid.json` 是占位路径示例，默认 `backend: replay`，不会联网。
复制到 `configs/jev.local.json` 后填写模型、数据和独立输出目录。

- `reward: hybrid`：原奖励 / `base_scale` + `weight` × 评审奖励。
- `reward: decisions`：仅评审奖励，不加载 LPIPS。
- `decisions.base_reward`：R0 或 R1；R1 还需原配置的 `reward_scales`。
- `decisions.backend`：`replay` 读取评分回执，`api` 为缺失评分调用官方 API。
- `decisions.model`：API 模式目前仅允许 `gpt-6-luna`。
- `decisions.review_dir`：建议放在被忽略的 `decision_reviews/` 下。
- `decisions.max_api_calls`：每次进程允许的总 HTTP 尝试次数，包含重试，默认 4。
- `decisions.timeout_seconds`：单次请求超时，默认 60 秒。
- `decisions.retries`：429/部分 5xx/传输错误的额外尝试数，默认 1。
- `base_scale` / `weight`：必须有限且大于零；示例 1.0 / 0.25 尚未校准。
- `paper_locked` 必须为 false；当前仅支持单进程，避免评分失败时 DDP 等待。

API 使用 `OPENAI_API_KEY` 环境变量，仅请求固定地址
`https://api.openai.com/v1/decisions`，拒绝重定向，不提供第三方自动回退。
密钥不得写入 JSON 配置、仓库或日志。HTTP 认证/权限错误立即停止；失败和拒答不会记为零分。
调用上限是请求数上限，不是美元预算。恢复启动会重置计数，已缓存评分不会重复调用。

```bash
USE_TF=0 USE_FLAX=0 PYTHONPATH=src "$PYTHON" -m futureworlds train --config configs/jev.local.json
```

模型、codec、动作预处理和 T5 必须匹配。原搜索、候选历史、组内优势和策略更新保持原算法。
API 提供的连续 score 除以最高等级索引后归一化到 [0,1]，再对两个评分项取均值，
不会先取 argmax 丢弃分数精度。完整概率分布和 confidence 被记录；confidence 不参与奖励。

## 网络与环境

本机 Clash Verge 当前混合端口为 `127.0.0.1:7897`。单步实验使用临时 SSH 隧道：

```bash
# 本机运行；远端代理只绑定服务器回环地址
ssh -N -T -o ExitOnForwardFailure=yes -p 8081 \
  -R 127.0.0.1:17897:127.0.0.1:7897 root@10.20.0.71
```

在需要 API 的服务器进程中设置 `HTTPS_PROXY`、`HTTP_PROXY` 为
`http://127.0.0.1:17897`。本机或隧道离线会导致 API 不可达；这不是持久服务。
勿把官方 API key 配置到中转站。运行命令中不直接写密钥，可用 `getpass` 读取再设置环境。

CPU 回归测试环境：本仓库 `.venv/bin/python`，通过 `.pth` 复用 RoboCasa CPU 依赖。
GPU 联调复用 `/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python`，
运行时 `PYTHONPATH=src` 指向本仓库，不修改另一项目代码或环境。

## 评审材料和回执

```text
decision_reviews/<request_id>/
  request.json         # 任务、模型、rubric、帧索引、数组校验值、评分协议版本
  frames.npz           # initial_rgb、prediction_rgb、raw_actions
  score.json           # 评分来源、官方响应、传输信息
  accepted_score.json  # 校验后锁定的回执
```

这套文件是本项目内部格式，不是 OpenAI HTTP 请求格式。
哈希绑定视觉内容、动作、任务、dataset、rubric、model、预处理和评分协议。
发送前验证数组与元数据哈希；恢复时不允许已读取回执原地修改。
API 响应必须完整匹配模型、问题名、等级标签、合法有限分数和概率分布。

RGB 为 uint8 THWC，初始观测索引 0、1；预测索引 2..H+1。
动作是数据集原始数值，不是 action token。物理动作语义未验证，所以当前
rubric 只评 `object_consistency` 与 `temporal_interaction`，不评动作一致性。
API 请求里也明确这一点，不把“任务成功”作为预测正确的判据。

旧人工回放协议仍支持 labels，每项 `incorrect/minor_error/consistent` 对应 0/0.5/1；
API 模式拒绝这种回执，必须使用 `official_response`。人工试验请使用明确的人工模型标识。

回放模式缺少评分会先导出该组所有候选，然后停止，不执行当前 optimizer step。
首次收集后应使用新 output 路径重新启动，保留 review_dir；有 checkpoint 时按原协议恢复。
纯评审奖励的整批候选同分时，原训练器会报错；不人工注入随机奖励。

## 验证与解释边界

```bash
USE_TF=0 USE_FLAX=0 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/verify_sources.py
```

单元和小模型测试覆盖：连续分数、拒答与非法响应、错误重试上限、未来 GT 隔离、
内容寻址与篡改拒绝、两种奖励驱动参数更新、缺少评分不执行 optimizer step。
合成测试中的人工分数明确标为测试，不进入实际训练。

真实单步试验配置位于 `configs/jev-smoke.local.json`，输出
`runs/jev_api_smoke_20261012`。这是 4 条候选、预测 4 帧、1 次优化器更新的链路验证，
不是论文结果，也不能证明效果优于 baseline。实际结果另见本目录中的运行报告。

后续先做固定候选人工对照、评审稳定性和同分率校准，再比较原奖励、纯评审奖励、
混合奖励。最后用独立人工评估与 PSNR/SSIM/LPIPS 评价，不能单凭评审自己的分数上涨。

## 短程试验结果

见 [校准与对照报告](JEV_PILOT_RESULTS_20261012.md) 和 [实验协议](JEV_PILOT_PROTOCOL.md)。
官方50RPD限制阻断20步混合训练；已完成缓存重放的两步匹配对照，不应将其写成20步试验。
