# E1 修正、混合 codec 与 E2 实验

服务器根目录：`/media/tyshao/fail_WM/FutureWorlds_SFT`。

## 已完成的修正

旧报告 `evals_diagnostics_20261009/RESULTS_zh.md` 已撤回用末态 5.78px 阈值淡化乐观的结论。旧表格仅保留为审计输出。重新解码旧 36 条 MP4 后，混合模型 b03/b04 的采样分别有 2/8、4/8 条更靠近目标；b03 beam 的末帧未检出方块。结果见 `evals_e1_expanded_20261009/old_e1_spatial_audit.json`。

这些计数是图像位置代理，不等于已经验证的物理成功率。先前“45条真实代理失败经codec变成7条代理成功”的结果也只能解释为重建后代理判定翻转，不能直接称为16%的物理乐观率。E0失败事件窗口NLL较高支持优先检查这些窗口，但不同窗口的视觉复杂度不同，不能仅凭均值差证明误差集中机制或teacher forcing因果关系。

## 扩展 E1：已启动，服务器后台继续

- 固定 10 个验证初态组，每组 2 成功 + 2 失败，共 40 条；两个模型，各 beam4 + 8 次采样，共 720 条预测。选择在生成结果前冻结，`selection.json` 保留来源、初态及几何信息。
- 失败案例优先选择真实末态离目标和起点都至少 8px 的，再优先 dropped/off-counter，并让同组第二条尽量覆盖另一失败类型。最终20条失败中11条满足三点分离。object_off_counter 可能出视野，末态不可见时保留缺失，不能强行推断坐标。
- 三个参照：真实末帧方块位置、真实目标位置、真实首帧方块位置。报告原始连续距离和最近参照类别，不用成功距离阈值。末帧未检出单列；相距不足5px的参照或距离差不足2px的结果标为歧义，离所有参照超过10px的结果单列。
- 同时输出“偏向目标”代理比例：真实失败与目标、起点与目标都至少8px，预测离目标比另两点近。报告0/2/4px裕量敏感性，含未检出的上下界及按初态组的比例。未检出不当作正确失败。
- 全程跟踪用于区分真实停留、预测整体不动和错位；不能把最近起点直接称为没动。成功控制结果单列。
- 原始生成帧直接计算指标，无损保存末帧 PNG；MP4 编码后再解码检查帧数和重算指标，两套指标都保留。未保存大量预测 NPZ。
- 同一组分支和8次采样并不独立，不把720个输出当作720个独立实验。指标尚未经过生成视频人工标注验证，不宣称真实成功判断准确率。

查看结果：

```text
evals_e1_expanded_20261009/REPORT_zh.md
evals_e1_expanded_20261009/summary.json
evals_e1_expanded_20261009/run.log
evals_e1_expanded_20261009/videos/{success,natural}/{case}/last_frames.png
```

`complete_success_natural.json` 出现才表示两个模型的40条案例全部完成。报告每案例完成后更新。开始时每个模型×案例约160秒，预计约3.5–4小时，取决于轨迹长度和共享GPU负载。中断后续跑：

```bash
cd /media/tyshao/fail_WM/FutureWorlds_SFT
/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python start_expanded_e1.py
```

新WM bundle可复用案例选择：`expanded_e1.py run --model 新标签=/绝对路径/bundle`。新codec不能直接替换进旧WM就当公平对照：需重新编码cache并在固定WM协议下训练，再做对比。

## 8卡 codec：可提交

```bash
bash /media/tyshao/fail_WM/FutureWorlds_SFT/run_mixed_codec_8gpu.sh
```

入口一次完成训练、验证、checkpoint选择和通过选择条件后的独立bundle安装，支持从自己的checkpoint续训。所有环境、模型、数据路径均在 `/media`。

- 原始800条训练数据：443成功+357失败，无验证/test混入，也不加入后来新增失败，以单独检验原数据配比改动。
- 从原始RT-1 tokenizer开始，保持旧success codec的seed、50000步、global batch32、学习率5e-5、warmup5000、10000步后原有GAN分支、损失和增强流程。每2000步保存并验证。本次不引入ROI训练loss，不修改原训练入口。
- 100条验证轨迹，事件前2至后4帧和末帧；成功/失败分别报告方块ROI MAE、中心误差、漏检率，以及末态偏移/缺失。记录每条轨迹指标，保留全图指标。
- 固定选择规则：失败 `ROI_MAE_255 + 32 × miss_rate` 越低越好；成功ROI不超过旧已部署codec的1.10倍+0.25，成功漏检率增幅不超过1个百分点。权重32是预先设定的惩罚系数，不是物理常量；完整候选指标均保存，不能只看单个综合分数。
- 所选模型还必须改善旧已部署codec的失败综合分数；没有通过时，输出 `NO_CHECKPOINT_IMPROVES_WITHIN_SUCCESS_GUARD`，不自动推广新bundle。
- 旧模型比较基准为实际已安装的 `assets/futureworlds_bundle_robocasa_codec_10k/robocasa/codec`，不是按文件名猜测步数。其manifest记录实际来自旧codec的step044000。
- 输出：`runs/paper_adapt_mixed_codec_20261009/`；选模记录 `selection.json`；通过后bundle为 `assets/futureworlds_bundle_mixed_codec_20261009/`。

已通过新指标回归检查，单卡smoke覆盖生成器一步、判别器一步和成功/失败ROI验证；smoke日志为 `experiments_20261009/codec_smoke.log`。单卡smoke提前启动未训练判别器，仅验证执行路径，不能解释其中GAN/gradient penalty数值为正式收敛指标。未运行真正8卡DDP，需专用训练机提交。

## E2：基线完成后提交，两种SFT顺序执行

```bash
bash /media/tyshao/fail_WM/FutureWorlds_SFT/run_e2_events_8gpu.sh
```

入口检查扩展E1完成标记后，先success、后natural，共两次10000步训练；现在不会提前开始E2。基线完成不代表已人工复核所有视频，训练结果仍须用固定评估对照。

- 固定现有codec，复用其已验证cache。与原两个10k基线相同RT-1初始化，而非从训练后checkpoint再加训。
- 10000步、global batch64、每卡batch8、原学习率和warmup、每2000步保存checkpoint，损失函数不变。
- 逐cursor完全复用原Schedule的轨迹ID和原target；独立随机数以50%概率把target换成该轨迹事件窗口内均匀采样。成功/失败同规则。无事件则回退原target。
- 事件：contact/grasp/lift/drop/release/grasp_attempt_failed；`ceil(state_index/4)`，窗口[-2,+4]，target≥2。
- 每个训练分支640000条schedule已生成并检查ID/target范围。混合组成功样本354628、失败285372，与原Schedule逐cursor一致。成功事件target占比由35.8%增至67.9%；失败由22.8%增至61.4%。对称的是采样规则，不强行令两类实际事件占比相等。
- 配置/协议/schedule：`experiments_20261009/e2_{success,natural}*.json`；独立输出 `runs/{success_only,natural}_sft_e2_events_10k_20261009/`。原训练代码、checkpoint、cache不覆盖。

这次E2与混合codec是两个独立对照，避免一次改动两项而无法解释改进来源。新codec通过局部验证后，需按原WM训练设置建立独立对照；再考虑叠加E2。
