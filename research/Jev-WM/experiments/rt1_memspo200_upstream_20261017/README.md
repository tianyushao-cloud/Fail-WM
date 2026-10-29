# RT-1 MemSPO200：按公开仓库配置准备

来源：https://github.com/Alexander-wu/FutureWorlds
固定提交：7b8f26a5d165c58645212093d2e0dfffc4278643

本目录source/由该Git提交直接导出，未使用本地工作区或vendor里的修改版。
配置来自configs/train/rt1_memspo.json。训练参数逐项保持一致：
200步、global batch16、seed20260915、horizon10、lr1e-6、R0、clip0.2、KL0.001、diversity0.1，每50步保存。
adapter_lr=1e-4和warmup_steps=500保留在JSON中，但原训练器只在SFT分支使用；RL的backbone与adapter统一固定lr1e-6。
RT-1主表另有500步配方，本轮使用已确认的200步匹配对照。

当前入口使用用户指定的4卡；原仓库工作流示例为8卡。沿用原生DDP训练器，每卡累积4个输入窗口（每窗口生成4个候选），保持global batch16。
启动器禁止单卡回退；当前SSH只可见1卡，尚未启动此方案。

## 命令
在能看到4张GPU且共享/media/tyshao目录的训练平台执行：
bash /media/tyshao/Jev-WM/experiments/rt1_memspo200_upstream_20261017/run_4gpu.sh

仅核验配置/资产，不进行GPU计算：
bash /media/tyshao/Jev-WM/experiments/rt1_memspo200_upstream_20261017/run_4gpu.sh check

完整流程：配置/源码/权重核验→4卡特征缓存→独立4卡global batch16单步检查→从SFT重新开始200步训练→导出→同协议评测SFT和RL→comparison.json与REPORT.md。
固定最终step200，不按测试结果挑选checkpoint。断点恢复保持卡数不变。有完整checkpoint时同一命令恢复；首次checkpoint前失败需要单独处理失败输出，不能假装已续训。

## 数据适配
使用官方RT-1原始数据构造的本地划分：512训练、32验证、64测试，episode互不重叠。
每episode在准备阶段以固定种子选择一个合法34帧窗口；本轮采样顺序使用上游Schedule类、seed20260915，生成3200项新窗口schedule。
公开仓库只提供原始固定schedule及cohort IDs，没有发布原窗口生成清单。训练器读取清单中的窗口，不在train循环内重新随机裁剪。
因此保留原方法/超参数，改用本地固定数据清单，paper_locked=false。随机起点在准备阶段固定，SFT与RL评测复用完全相同窗口。
原始SFT训练集重合未知。图像指标不代表任务成功率。

## 帧数核查
远端公开configs/train/rt1_memspo.json明确horizon=10。
src/futureworlds/training.py读取horizon+2张图，即2张条件图+10张奖励真值图。
configs/recipes/matched-200.json和workflow.py评测10/20/32帧，逐帧Beam4。
pipeline.py中的8是codec编码分块大小，不能作为RL rollout长度。
RLVR-World上游另有segment_length=8的配置，但不能拿它替代此FutureWorlds配置。

## MemSPO实现
1. 从SFT复制actor和冻结reference；冻结codec/T5，更新actor主干与文本adapter。
2. 同一窗口以4组×2beam产生候选；跨帧保留搜索分支，每组最终返回1条视频，共4条。
3. 每条分支保留自己的历史。输入由首帧场景token、固定第2帧anchor、最近6个状态及对应动作组成；每帧重建KV和位置。
4. 跨组同一位置token频率惩罚（0.1）促进候选差异，只影响搜索分数；训练评分使用原模型logprob。
5. 将4条视频解码，逐帧计算与GT的MAE和LPIPS_VGG，奖励=-mean_t(MAE+LPIPS)。
6. 同一窗口4条奖励组内标准化为advantage；同一条视频的全部生成token共享该视频advantage。
7. old/current/reference对候选使用一致历史计算token logprob，PPO式clip目标+0.001参考模型KL，梯度范数裁剪1.0，AdamW一步更新。
8. 每global batch重新构造候选并只做一次optimizer.step，没有多轮PPO重放；old/current在更新前一致，clip并非多epoch训练时那样持续触发。
9. 确定性搜索候选不等于on-policy采样；这是搜索引导的代理优化，仓库也明确标记on_policy_grpo=false。

文件：recipe_audit.json（参数差异审计）、source_hashes.json、preparation_check.json、train.json、schedule.json、run.py。

## 4卡核验
run_4gpu.sh check只核验配置、源码及权重，不启动CUDA训练。核验结果保存于preparation_check.json及preparation_check_4gpu.json。
4卡真实运行尚未执行；启动后先做独立单步分布式检查，再从SFT开始正式200步。保留run_8gpu.sh供原8卡入口使用，两者共用输出目录，不可在已开始的实验中切换卡数。
