# 原始首两帧条件的本地成功数据 MemSPO

训练平台8卡前台启动：
```bash
cd /media/tyshao/Jev-WM/FutureWorlds && exec bash scripts/run_local_success_8gpu.sh all
```

all依次执行：SFT/codec来源及划分审核、原始起点数据准备、8卡预检、SFT验证/测试、200步RL、导出RL、同协议RL验证/测试、生成REPORT.md。启动命令保持前台，所有输出经tee写入logs/all.log。

初始化为 assets/local_success_sft：RT-1后在443条本地成功轨迹上SFT10000步，配套success-only codec step044000。新的RL从此SFT重新开始，不加载之前中间起点RL的checkpoint或优化器。训练443成功、失败0；val53成功/test59成功，各10个初态组，按episode和初态与训练分离。

训练和评估均以原始轨迹的images[0:2]作为唯一初始真实观测，source_start=0。每条轨迹仅准备一个从第0帧开始的42帧文件，动作维数52，原始动作顺序不变，末帧未使用动作置0。训练读取首12帧，产生10帧RL rollout；评估读取首34帧，产生32帧Beam4 rollout。不在中间重设参照，不注入未来GT。

固定学长发布的MemSPO/R0：200步、global batch16、seed20260918、horizon10、统一lr1e-6（含文本适配器；配置adapter_lr字段仅SFT使用）、clip0.2、KL0.001、diversity0.1、4组×2beam、每50步保存；最终200步对比。原Schedule保留task/episode采样，每episode只有一个起点0窗口。冻结codec/T5/reference，原vendor训练器、奖励与目标保持hash锁定。R0=-mean(MAE+LPIPS_VGG)，不调用Decisions。

这轮RL优化的是轨迹开头的10帧，不是整段任务完成质量；评估32帧也可能尚未到任务结束。不要把指标改善解释为任务成功率。

输出统一复用 runs/local_success_memspo，没有v编号：protocol.json、schedule.json、train.json、sft_data_audit.json、data_audit.json、data/{train,validation,test}/manifest.json、train/checkpoints/step_000200、evaluation/{validation,test}/{sft,rl}、REPORT.md、comparison.json。正式8卡日志 logs/all.log；已完成阶段按原指纹检查后跳过。正式训练后不允许改变卡数续训。

原首两帧数据审计 preparation_audit.json 逐条确认准备数据与原episode开头相同，所有source_start=0且无失败。smoke/仅用于单卡接口检查，不是正式训练结果。

2026-10-15按用户指示删除Jev-WM中的4个中间/居中起点MemSPO实验目录：local_success_memspo（已由本轮重建）、scholar_success_memspo、success_memspo_4gpu、success_memspo。删除内容含派生数据、RL checkpoint、评估视频、临时诊断；记录 docs/INITIAL_REFERENCE_RESET.json。原始fail_WM数据、旧预览、SFT/codec和assets/local_success_sft保留。旧中间起点实验不能用新入口续训，程序会拒绝。

已有decoding FP32求和校验兼容修复保留在 vendor/scholar_futureworlds/COMPATIBILITY_PATCHES.json；不改变候选排序或训练奖励。验证集曾用于codec选择，val/test被查看过，本轮不是新盲测。


训练器核验：MemSPO使用统一固定lr=1e-6，warmup_steps=500只在SFT分支生效。本轮并非200步仍处于warmup。原始首两帧+horizon10只直接奖励原帧2..11；事件审计显示443训练轨迹均未覆盖contact/grasp/lift。原因分析见 runs/local_success_memspo/diagnosis/final_review/CAUSE_ANALYSIS.md。
