# 世界模型诊断结果（2026-10-09）

本次完成 E0 真实帧似然与动作交换、tokenizer 局部重建审计，以及小规模 E1 beam4/随机采样对比。没有进行新训练。全部数据和视频保存在服务器。

## 主要结论

1. 混合失败数据不是没有作用：混合模型对真实失败事件窗口的平均 token NLL 更低，且更经常偏好与失败画面正确配对的动作。但似然改善不等于开放循环预测准确率已经提高。
2. tokenizer 存在具体的局部问题：用真实 token 重建时，部分方块会偏移、变淡或消失。45 条真实视频代理判为失败的轨迹，仅 codec 重建就使其中 7 条变成代理成功、5 条变成不可判。失败轨迹方块区域误差更高。不能用全图 PSNR 很好来排除这一问题，也不能据此把全部 WM 错误归因于 tokenizer。
3. 【修正】原 E1 成功距离阈值不能识别“真实留在起点，但预测已运到目标旁”的乐观偏移；代理 failure 不等于正确预测失败。用户逐帧末态复核指出混合模型 16 条采样中约 6 条偏向目标，另有物体不可见案例；该计数保留为用户复核记录，新的逐帧距离审计见 evals_e1_expanded_20261009。原报告据阈值结果淡化乐观问题的表述撤回。natural/g00006_b03 的 beam4 最后一帧方块不可见，应按末帧不可判处理，不能用最后五帧中位数的 failure 代替。
4. 当前实验没有证实 teacher forcing 是主因，没有证明动作条件失效，也没有测出生成视频的可靠成功/失败分类准确率。

## 范围与模型

- 验证集 100 条：53 成功、47 失败，来自 10 个初态组。分支相关，不能视作 100 个独立场景。
- 两个模型均为自己适配 tokenizer 后训练的 10,000 步模型，token cache 版本；本次不是学长权重对比。
- success：`runs/success_only_sft_new_codec_10k/checkpoints/step_010000`。
- natural：`runs/natural_sft_new_codec_10k/checkpoints/step_010000`。
- 两者使用 `assets/futureworlds_bundle_robocasa_codec_10k`；诊断包在本目录的 `success_bundle/` 和 `natural_bundle/`。
- 原混合训练集是 443 成功 + 357 失败；不要把本结果解释为 70% 失败版本的结果。

## E0：真实下一帧似然

每模型评分 2,269 个 target。事件帧采用 ceil(raw state_index / 4)，覆盖事件前 2 帧至后 4 帧；另外每条轨迹选 4 个非事件 target。使用真实历史进行 teacher forcing。下表先在每条 episode 内平均，再对 episode 平均；NLL 单位为 nat/token，越低越好。

| 数据 / target | 仅成功模型 | 混合模型 |
|---|---:|---:|
| 成功 / 事件窗口 | 1.0739 | 0.7500 |
| 成功 / 非事件 | 1.0946 | 0.7174 |
| 失败 / 事件窗口 | 1.7447 | 1.0484 |
| 失败 / 非事件 | 1.6553 | 0.8428 |

这里比较同一批真实 target 在两个模型下的 NLL；不把成功视频与失败视频各自的平均 NLL 当成成功概率。表中为 4,375 个视觉 token 内归一化的 NLL，原始 JSON 也保留完整训练词表 NLL。事件后历史可能已暴露结果，所以这项检查不能单独证明模型能在事件发生前预测失败。

评分实现与原始 sft_example + model.loss 对照：训练 loss=1.139909625，诊断完整词表 NLL=1.139909506，绝对差=1.19e-07，通过 1e-4 容差。

## E0：动作交换

同一物理初态配对；统一首帧编码上下文。仅保留初始两帧像素差极小、所有保留历史帧 MAE ≤ 0.5/255 且动态 token 一致率 ≥ 95% 的窗口。用相同成功分支历史，对成功/失败 target × 成功/失败动作做 2×2 评分。47 对中 44 对通过筛选，共 81 个窗口，79 个窗口的成功/失败 target token 不同。窗口数量选择不依赖模型分数。

| 配对指标 | 仅成功模型 | 混合模型 |
|---|---:|---:|
| 失败 target 偏好自身动作（对/44） | 26 | 37 |
| 成功和失败 target 均偏好自身动作（对/44） | 24 | 33 |
| 2×2 配对选择性为正（对/44） | 40 | 43 |
| 失败动作偏好均值（nat/token） | 0.0798 | 0.1326 |
| 2×2 配对选择性均值（nat/token） | 0.2482 | 0.2475 |

失败动作偏好 = NLL(失败 target | 成功动作) − NLL(失败 target | 失败动作)。2×2 选择性是成功、失败各自动作偏好的和。正值支持正确配对。混合模型在更多配对上方向正确，但 2×2 选择性均值没有增加，不能笼统说动作敏感度全面增强。这些比例不是成功/失败分类准确率；历史容差也不是物理等价证明。

## Tokenizer：真实帧重建审计

总计 1,969 帧（事件窗口及末帧）。ROI 为真实红色方块包围框外扩 8 像素；真实方块不可见时 ROI 缺失，不计零误差。位置误差仅统计两边均可检测的帧；另报重建中漏检数量。

| 真实数据 | 轨迹数 | 帧数 | 方块 ROI MAE（0–255） | 可见方块中心误差（像素） | 真实可见但重建漏检帧数 |
|---|---:|---:|---:|---:|---:|
| success | 53 | 1233 | 8.990 | 0.642 | 12 |
| failure | 47 | 736 | 11.843 | 3.420 | 30 |
| contact_without_lift | 9 | 158 | 15.486 | 6.670 | 15 |
| dropped_before_goal | 7 | 144 | 14.303 | 2.078 | 6 |
| grasp_miss | 15 | 120 | 8.630 | 3.992 | 2 |
| lifted_but_not_placed | 15 | 299 | 11.637 | 1.485 | 7 |
| object_off_counter | 1 | 15 | 13.105 | 4.031 | 0 |

可直接看 `codec_worst_00.png`（方块偏移）、`codec_worst_03.png`（真实可见方块在重建中消失）、`codec_worst_08.png`（方块变淡）。这些是有意挑出的高误差例子，不代表所有帧都如此。

另外重建全部 100 条的最后 5 帧，仅检查 codec 是否改变同一 RGB 末态代理判定：

| GT | 真实视频代理判定 | codec 重建代理判定 | 条数 |
|---|---|---|---:|
| 失败 | failure | failure | 33 |
| 失败 | failure | indeterminate | 5 |
| 失败 | failure | success | 7 |
| 失败 | indeterminate | indeterminate | 1 |
| 失败 | success | success | 1 |
| 成功 | success | success | 53 |

## E1：旧阈值实验（仅留作审计，结论已修正）

以下表格是旧版最后五帧阈值输出，不能据此计算真实失败预测能力、也不能排除乐观。仅成功模型可能存在方块不动/错位，必须加入成功案例和运动轨迹对照。后续报告采用末帧与真实位置、目标、起点的连续距离，并显式保留不可见和参照重合的不确定性。

固定选取验证集最先出现的 grasp_miss（g00006_b03）和 dropped_before_goal（g00006_b04），选择发生在查看 E0 分数之前。两条都来自同一个初态组；本实验不是总体评估。每模型每案例 beam4 一条，temperature=1 采样八条，共 36 条生成视频。初始仅给两帧真实画面，之后自回归至真实 episode 结束。采样分两批，每批四条，种子为 2026100900、2026100901，两个模型一致。

| 模型 | 真实失败案例 | beam4 代理判定 | 8 次采样：失败 / 成功 / 不可判 |
|---|---|---|---|
| success | g00006_b03 (grasp_miss) | failure | 8 / 0 / 0 |
| success | g00006_b04 (dropped_before_goal) | failure | 8 / 0 / 0 |
| natural | g00006_b03 (grasp_miss) | failure | 5 / 1 / 2 |
| natural | g00006_b04 (dropped_before_goal) | failure | 6 / 1 / 1 |

代理判定：最后五帧至少三帧可检测，红色面积中位数 ≥ 8、绿色面积中位数 ≥ 20，方块和目标二维中心距离中位数 < 5.7807 像素则暂记成功，否则失败；不满足可检测条件记不可判。阈值在同一验证集真实视频上拟合，其拟合分数不能作为独立准确率，更不能迁移成生成视频准确率。

该判定不能验证抓取过程、接触、物理合理性，也不能把物体消失算成真实失败。混合模型 grasp_miss 的 beam 末态方块很弱，判定尤其需要保留意见。采样中的“失败”只代表末态代理结果，不代表正确复现失败过程。

## 视频完整性与轨迹检查

已逐条解码 36 条 MP4，检查帧数与真实未来帧数一致。编码 CRF=14，5 fps，分辨率 320×256。下表位置误差按视频先平均再平均，只统计红色方块两边可见帧；漏检单列，避免用缺失降低误差。

| 模型 | 解码 | 视频数 | 可见帧平均方块位置误差（像素） | 比较帧数 | 真实可见但预测漏检帧数 |
|---|---|---:|---:|---:|---:|
| success | beam4 | 2 | 7.578 | 106 | 8 |
| success | sampling | 16 | 10.409 | 853 | 59 |
| natural | beam4 | 2 | 10.424 | 96 | 18 |
| natural | sampling | 16 | 8.045 | 801 | 111 |

以上来自解码后的 MP4 和颜色检测，仅为轨迹误差代理。不能由末态失败比例推断校准概率或声称模型已正确理解失败。

## 下一步建议

1. 先扩展到不同初态的成功/失败配对，人工核查一小批生成视频，区分正确成功、正确失败、过程不对和画面不可判；固定验证指标后再看独立 test。
2. 固定当前 tokenizer 和解码方式，做 E2 事件窗口采样对照；成功与失败对称过采样，保持训练步数、有效 batch、基座、验证样本一致。当前证据支持先检验训练覆盖，而不是直接宣布 teacher forcing 已被证实为主因。
3. 优先安排独立的 tokenizer 成功+失败适配对照，以失败方块 ROI、漏检率及重建前后末态一致性选择 checkpoint。更换 tokenizer 后重新建 token cache，并检查 WM 与新 token 编码的兼容性；不要把换 codec 和换 WM 目标混在一次实验里。若训练资源有限，先处理这个已观察到的缺陷，再投入大规模 WM 训练。
4. E4 的 8×10 区域权重只是空间代理，需要核实动态 token 与图像局部的对应关系后再单独消融。E5/E6 暂后置，动作交换结果并不支持“动作条件基本没起作用”。

## 文件与复跑

- 本报告：`RESULTS_zh.md`；原始汇总：`REPORT.md`。
- E0：`e0_success.json`、`e0_natural.json`、`e0_pair_selection.json`。
- Codec：`codec_audit.json`、`codec_worst_*.png`。
- 评分/末态重建校验：`verification.json`。
- E1：`e1_summary.json`、`sampling_tracking_audit.json`。
- 视频：`e1/{success,natural}/{g00006_b03,g00006_b04}/`，各含 `real.mp4`、`beam4.mp4`、`sample_00.mp4` 至 `sample_07.mp4` 和 `last_frames.png`。
- 诊断脚本：项目根目录 `diagnose_world_model.py`；报告脚本：`write_diagnostic_report.py`。
- `provenance.json` 记录数据与 bundle manifest 校验和，`verification.json` 记录最终诊断脚本校验和。

```bash
cd /media/tyshao/fail_WM/FutureWorlds_SFT
# PHASE 可取 prepare / pairs / e0 / e1 / verify / sampling_audit / summarize
env PYTHONPATH=/media/tyshao/fail_WM/FutureWorlds_workspace/code/src USE_TF=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  /media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python diagnose_world_model.py PHASE
python3 write_diagnostic_report.py
```

重跑同一阶段会替换本诊断目录内对应阶段的产物；本次没有修改训练器、原始数据和已有训练 checkpoint。
