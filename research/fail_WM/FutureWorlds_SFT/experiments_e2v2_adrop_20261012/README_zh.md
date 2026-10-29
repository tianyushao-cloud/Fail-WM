# codec34k：E2v2 × action dropout

状态：准备完成，待提交到 8 卡训练机。B/C/D 正式训练尚未启动。

提交一次命令：

```bash
bash /media/tyshao/fail_WM/FutureWorlds_SFT/run_2x2_8gpu_20261012.sh
```

使用挂载目录里的现有 Python 环境及全部资源。需要 8 张可见 GPU；每卡训练 batch=8，总 batch=64。

|组|target采样|动作dropout|训练输出（相对SFT根目录）|
|---|---|---|---|
|A|原均匀|无|已有 `runs/natural_sft_mixed_codec34000_10k_20261011`|
|B|50%原均匀 + 50%事件/阶段窗口|无|`runs/natural_c34_e2v2_10k_20261012`|
|C|原均匀|25%样本全部prefix动作替换为9006|`runs/natural_c34_adrop25_10k_20261012`|
|D|与B相同|与C相同|`runs/natural_c34_e2v2_adrop25_10k_20261012`|

三组均从原 RT-1 初始化，各 10,000 步，每 2,000 步保存，复用 codec34k 的800条混合数据cache（443成功、357失败）。dropout按cursor确定，C/D使用相同掩码；评估正常输入动作。新入口在SFT根目录，原futureworlds源码/旧cache未改。

开训门槛按Opus 11:50修订为2.5倍：关键202条失败的L..L+6曝光18991→47483（2.5003倍），保留p=0.5。`protocol.json`为当前冻结协议；`preflight.json`保留原3倍门槛检查的历史结果，不是当前启动状态。

脚本自动依次完成：

1. 校验源码、权重、manifest、800条cache、数据隔离及共享LPIPS依赖；运行3步单卡smoke。
2. 8卡依次训练B、C、D，并校验最终checkpoint、10,000行日志和实际dropout计数。
3. 导出B/C/D及用于帧指标对照的A bundle。
4. 8个分片运行冻结51例事件评估，每例start/+2，beam4+8次采样；分片与GPU一一绑定。
5. 在100条val上计算A/B/C/D的固定horizon10/20帧指标。LPIPS@20按原评估器定义为前20个生成帧的平均。
6. 复用Opus的A事件结果，汇总主指标、初态组bootstrap和护栏。只报告数字，test集不使用。

整个作业支持再次提交同一命令恢复。完整checkpoint保留；中断后的日志尾部和未完成checkpoint移入`experiments_e2v2_adrop_20261012/interrupted/`。若首次checkpoint之前中断，旧运行目录先归档再从同一初始化重启。已发布checkpoint校验失败、协议改变会报错，不会默默重训。

逐案例评估可以续跑。若最后A事件结果仍不齐，会保存`comparison_pending.json`并停止；A完成后重新提交同一命令即可。此文件是历史等待记录，最终状态以`comparison.json`为准。

日志：`experiments_e2v2_adrop_20261012/logs/job_*.log`。

结果：`evals_2x2_20261012/COMPARISON_zh.md`和`comparison.json`。

准备阶段已通过真实缓存样本的3步GPU更新、null动作位置/标签/非动作输入检查、null嵌入梯度检查、CPU断点恢复和实际训练器fingerprint对齐检查、LPIPS/SSIM依赖检查。单卡smoke每步2个样本，峰值5.24GiB；不等于已经执行8卡DDP，也不代表模型效果已验证。
