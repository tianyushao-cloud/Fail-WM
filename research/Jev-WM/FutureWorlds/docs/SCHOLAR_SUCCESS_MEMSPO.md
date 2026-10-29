> 2026-10-15：本文所述中间窗口实验产物已按用户要求删除。当前使用原始首两帧入口 scripts/run_local_success_8gpu.sh，见 LOCAL_SUCCESS_MEMSPO.md。以下仅为旧方案说明。

# 按发布源码进行成功数据 MemSPO

训练平台启动命令（前台运行）：

```bash
cd /media/tyshao/Jev-WM/FutureWorlds && exec bash scripts/run_scholar_success_4gpu.sh all
```

默认路径不变。原 RoboCasa SFT 初始化，4卡，200步，全局batch16，每卡累积4，seed20260918，horizon10，lr1e-6，clip0.2，KL0.001，diversity penalty0.1，4组×2beam。R0=-mean(MAE+LPIPS-VGG)。原 AdamW/组内标准化/KL/梯度裁剪逻辑，冻结 codec/T5/reference。无需重新SFT，不调用Decisions。

模型后端在 vendor/scholar_futureworlds，来自仓库提交7b8f26a5d165c58645212093d2e0dfffc4278643；SOURCE_LOCK.json记录源码/配置哈希，入口逐个核对。训练只加载该固定源码，不加载之前为Decisions修改的training.py。不影响旧实验。

严格沿用发布训练实现和 robocasa_memspo.json 的200步匹配消融超参数；主表400步是另一份配方。数据、窗口、顺序、4卡执行为本次指定适配，paper_locked=false，不能称为论文原始实验复现。

443条成功训练轨迹枚举全部合法42帧窗口起点，使用发布sampling.Schedule的task→episode→window抽样器生成3200次取样的固定schedule，仅保存实际选中的窗口。训练器按原代码读取其前12帧。不是居中裁切，不做阶段加权。原论文窗口起点生成规则没有恢复，合法起点枚举是公开的数据适配，不能保证与其未知规则一致。42帧窗口尾部不参与horizon10奖励，因此不要声称已覆盖全部末尾阶段。

val/test筛选53/59条成功轨迹，每条独立固定种子选一个合法42帧窗口，评估前34帧。保持原划分，检查episode和initial_state分离。SFT/RL同窗口同动作，2帧初始观测，Beam4，FP32，TF32关闭，ReCAP full，10/20/32帧，无GT refresh。与旧居中窗口实验不可直接比较均值；本次仅固定最后200步进行最终比较。既有测试集曾参与诊断，不能作新的盲测泛化证据。

all：数据准备→GPU/资产检查→SFT验证/测试→4卡RL→导出→RL验证/测试→REPORT.md。每50步保存，重复同命令可恢复最近完整checkpoint并复用有效评估。首次checkpoint前中断需新OUTPUT。

输出：runs/scholar_success_memspo/REPORT.md、comparison.json、protocol.json、recipe_comparison.json、data_audit.json、schedule.json、train/metrics.jsonl、train/checkpoints/、evaluation/。
日志：runs/scholar_success_memspo/logs/all.log，同时显示在平台控制台。
正式训练由用户在4卡训练机启动，当前服务器仅准备和验证。
