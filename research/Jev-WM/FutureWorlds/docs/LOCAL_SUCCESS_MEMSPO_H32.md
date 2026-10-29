# 32帧 MemSPO：原始首两帧条件

训练平台4卡前台命令：
```bash
cd /media/tyshao/Jev-WM/FutureWorlds && exec bash scripts/run_local_success_h32_4gpu.sh all
```

本轮从 assets/local_success_sft 原始SFT重新初始化；不是续训horizon10的RL。horizon32意味着输入原帧0/1，自主生成原帧2..33（共32帧），无GT refresh。443条成功训练轨迹、53成功val、59成功test，首帧参照始终为原轨迹第0帧。现有42帧输入文件足够。

训练与horizon10对照的唯一超参数差异为horizon=32：200优化步、global batch16（4卡每卡累积4条轨迹）、固定lr=1e-6（包含文本适配器）、KL0.001、clip0.2、diversity0.1、4组×2beam、R0=-mean_t(MAE+LPIPS_VGG)、seed20260918，checkpoint每50步。源码保持锁定。原RL分支不使用配置中的SFT warmup或独立adapter_lr。

训练时间范围改变使大多数抓取/抬升进入奖励；还不能覆盖大多数放置/完成事件。该实验改变了已发布配方的horizon，不声称超参数与论文全部一致。

all流程：数据/来源校验→4卡检查→独立4卡1步真实RL smoke（horizon32/global batch16）→SFT val/test→正式200步RL→50/100/150/200各自验证→按验证集32帧LPIPS最低选权重（完全相同则选较早步）→仅所选权重评估test→报告。

评估选择规则预先固定，checkpoint_selection.json记录全部候选与选中结果。validation_checkpoints/保留所有4步的val结果，rl_bundle和evaluation/validation/rl链接到选中权重及结果。与旧horizon10最终200步结果比较时，注明本轮验证选模；若要严格控制checkpoint步数，先比较两轮200步的验证结果。测试集不用于挑选checkpoint。

新输出 runs/local_success_memspo_h32/；旧 runs/local_success_memspo/完整保留。日志 logs/all.log；报告 REPORT.md；选择记录 checkpoint_selection.json。不是v编号迭代，目录后缀h32表示预测长度。

本机SSH服务器目前只有1张可见A100；smoke/是单卡功能/显存测试，distributed_smoke/是训练机执行的4卡真实更新测试。两者权重/优化器都与正式train/隔离。正式训练开始前需要在4卡平台执行上述命令。

可单独启动训练平台4卡检查：
```bash
bash scripts/run_local_success_h32_4gpu.sh smoke
```

断点重启用同一all命令，已完成阶段经指纹检查后跳过。请勿将OUTPUT指定为旧horizon10目录，程序会拒绝跨horizon续训。


已通过单A100真实32帧RL一步：4候选、奖励标准差0.02267、反向及AdamW完成；峰值allocated显存11.82GiB（reserved12.41GiB），该步计算73.90秒，不含加载/写盘。单卡global batch1，只验证功能与显存，不等于4卡实测吞吐。训练机all会独立执行global batch16的4卡一步检查。数据555条NPZ与原h10输入哈希一致，schedule完全相同，唯一训练超参数差异horizon10→32。
