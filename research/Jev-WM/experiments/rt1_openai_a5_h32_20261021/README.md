# RT-1 SFT → OpenAI Decisions 校准后多维奖励 RL（10 步）

本实验从 `/media/tyshao/Jev-WM/paper_assets/futureworlds_rt1` 的 `sft` variant 初始化。OpenAI Decisions 作为奖励裁判；训练对象是 FutureWorlds 世界模型。

基准为 `../rt1_qwen_a5_h32_20261020`，保留其 MemSPO、四组候选及搜索、32 帧 horizon、global batch 16、数据清单、schedule、seed、学习率、KL；奖励从原四维权重起步，按校准筛选规则决定纳入维度。仅将 Qwen 裁判换成 `POST https://api.openai.com/v1/decisions` 的 `gpt-6-luna`。本轮按用户要求训练 10 步，在第 5、10 步保存 checkpoint。当前一张可见 A100，累积 16 个窗口构成相同 global batch。

fidelity、motion、physics、task 的图像输入、提示词、两种视图和汇总函数沿用 Qwen 版本。API 返回的 0–4 级概率进入相同的期望分数和组内归一化。每轴 tau 用 Decisions 自身的视图差重新估计，不沿用 Qwen 的 tau。相同证据的多个评分问题合并到一个 HTTP 请求，所有原始响应、输入图片和校验信息保存在 `reviews/`。无图像指标奖励；GT 仅在 rollout 完成后提供给奖励端。

流程：4 个已有 train 窗口及原控制样本校准 → 单独 1 步训练检查 → 10 步正式趋势实验 → 第 5 步 validation → 第 10 步 validation/test → 自动比较报告。校准沿用各轴已有门槛；首次要求四轴全通过，完整校准后改用原 Qwen A5 的失败轴剔除规则，要求 fidelity/motion 通过；这是小规模试验，不等于完整裁判验证。校准失败或 API 失败会在 `status.json` 记录，模型不会使用伪造分数继续更新。

当前代理连接：服务器 `127.0.0.1:17897` 经 SSH 反向转发至用户本机 `127.0.0.1:7897`。API 阶段依赖本机代理和该 SSH 连接持续在线。key 从原有私有文件加载，不写入实验配置或日志。

```bash
# 本机建立转发（当前已由本轮任务建立；不要重复占用端口）
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:17897:127.0.0.1:7897 -p 8081 root@10.20.0.71

# 服务器中查看进度
cd /media/tyshao/Jev-WM/experiments/rt1_openai_a5_h32_20261021
cat status.json
tail -f logs/calibrate.log logs/pilot.log logs/train.log

# 单独离线检查，或重新启动中断的流水线
bash run_1gpu.sh check
bash run_1gpu.sh all
```

`preparation_check.json` 记录与 Qwen 基准的一致性核验。仅原有 `training.py` 增加奖励分支，新加 `openai_judge.py` 和 `openai_gt_reward.py`；推理、解码、搜索、数据、评测保持基准源码。旧 token cache 的指纹绑定旧源码，本实验重新编码相同冻结 codec/T5 特征。

首个真实 API 检查：一个 train 窗口上，codec GT 与黑屏的 fidelity 分数分别约 0.7625、0.1695，详见 `calibration/api_probe.json`。这仅说明接口与该明显错误检查有效。`REPORT.md` 与 `comparison.json` 在训练评测完成后生成，包含 SFT/第 5 步/第 10 步的验证趋势。最终 checkpoint 固定为第 10 步，不按测试结果选择。

## 断线修复与维度筛选记录

校准最初因 RemoteDisconnected 中断，303 份有效响应已缓存。已加入连接中断重试，测试增加到 6 项并通过；本机转发由临时 SSH 进程改为 launchd 服务 com.codex.jevwm.rt1openai.proxy，断开后自动重连。

4 个训练窗口的完整校准：fidelity/motion/task 的控制样本检查通过；physics 的 erase 检测率 0.50，低于原门槛 0.75。fidelity/motion/task 的有效组率分别为 0.25/0.75/0.50。初始四维整体门槛因此失败，原结果保存在 calibration/summary_initial_all4.json。

随后明确修改了首次整体门槛：按原 Qwen A5 已有规则剔除未过轴，原有每轴门槛、评分和 tau 不变。纳入 fidelity/motion/task，归一化权重为 0.384615/0.307692/0.307692。select_axes.py 可重现筛选，calibration/summary.json 记录 post_calibration=true 的修订及被剔除的 physics。本轮属于探索试验；与原 Qwen 四维结果直接比较，同时改变裁判和纳入维度，不是仅更换裁判的单变量消融。

本机停止本任务的自动重连代理服务可执行：launchctl bootout gui/501/com.codex.jevwm.rt1openai.proxy。

## 四卡运行

先在环境中分配并确认四张可见 GPU，再执行 run_4gpu.sh all。脚本在 GPU 数量检查通过后，仅停止本目录中当前单卡 launcher 的独立进程组，保留单卡 pilot/train 记录到带时间戳目录，再从同一 RT-1 SFT 起点运行四卡检查与 10 步训练。global batch 仍为 16，每卡累积 4 个窗口。校准和内容寻址 API 评分缓存复用。四卡断点可在同样四卡环境续跑；单卡 optimizer 断点不混用。当前一张可见 GPU 的环境会在停止旧任务前被拒绝。

后台命令：CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash run_4gpu.sh all > logs/all_4gpu.log 2>&1 &
查看日志：tail -f logs/all_4gpu.log

## 新容器凭据与代理修复

四卡 batch 容器不继承旧容器 /root 中的 key 与 loopback 代理。runtime.py 现优先读取 /media/tyshao/.config/jev-wm/openai.key（私有目录 700，文件 600），并在本容器无 17897 转发时使用共享私有配置中的认证代理。代理仅允许 CONNECT 到 api.openai.com:443，复用本机 7897 的 SSH 链路。命令和日志不包含凭据。四卡入口在移动输出目录之前检查凭据和代理 TCP 连通性。已模拟无本地转发的新容器，验证共享配置加载及认证 API HTTP 200。

## 四卡正式训练第 2 步的网络中断与恢复准备

四卡单步检查成功：optimizer_updated=true，16 组中 11 组有效，耗时 318.58 秒。正式训练第 1 步成功更新，耗时 317.19 秒；第 2 步在 2026-10-21 20:09 上海时间因 Decisions transport failed 停止。原设置每 5 步保存，故本次没有正式 checkpoint；原有日志与评分缓存保留，重启时未完成输出会归档。

已将 API workers 从每 rank 4 个降为每 rank 1 个（全局 4 个），请求超时从 90 秒增至 180 秒；代理 relay 的 I/O 与 idle 超时同步放宽并加入不含凭据的诊断。4 个此前失败的真实评分请求已成功重放，分别约 18.4/23.6/28.8/30.2 秒，官方响应已写入原缓存。

后续每步保存 checkpoint，interim_step 固定为 5，保留第 5/10 步评估。launcher 对新增日志中明确的 API 网络/限流/服务错误，最多再启动两次，每次等待 30 秒，并读取最后成功 checkpoint 续跑。认证、预算、评分校验或模型错误不自动重试。离线验证覆盖断线后恢复 checkpoint，以及旧日志网络错误不会使当前模型错误被重试。

本次修复不代表四卡任务已重新启动；需在四卡任务环境重新提交原来的 run_4gpu.sh all 命令。

## 提高评分并发

按用户优先速度的要求，后续 API workers 已从每 rank 1 调整为每 rank 8，四卡最多 32 个并发。共享请求间隔仍为 0.25 秒，单次请求超时 180 秒、网络重试、每步存档与自动恢复保留。run_4gpu.sh 会打印该并发配置。参数更改在下一次四卡任务提交时生效；没有修改模型、奖励维度、数据顺序和 10 步训练计划。

并发压测后的最终配置：每 rank 4 个 worker、四卡最多 16 个并发（相较之前全局 4 个提高 4 倍）。12 个此前失败的真实请求全部成功，约 32–84 秒；与先前 4 路约 18–30 秒相比，总吞吐提升有限，因此没有采用初步提出的全局 32。180 秒请求超时、每步保存和自动网络恢复保留。
