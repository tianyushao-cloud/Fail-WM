# 非方块复杂交互预览（2026-10-22）

服务器根目录：`/media/tyshao/fail_WM/RoboCasa/data/complex_nonblock_20261022`。

这批预览使用 RoboCasa Kitchen / PandaOmron / MuJoCo EGL。抽屉、导轨和收纳盒为程序化碰撞几何，饮料罐使用现有 robosuite CanObject 资产。均为自定义任务，不是官方完整厨房 benchmark。

| 目录 | 任务 | 关键交互 |
|---|---|---|
| `01_drawer_can` | 抓把手拉开抽屉，取出饮料罐并直立放到绿色目标，再关闭抽屉 | 滑动关节、开抽屉后取物、操作顺序、关抽屉 |
| `02_push_then_bin` | 沿导轨推动橙色圆饼到出口，再抓起放入绿色实体盒 | 平面接触推动、推抓切换、容纳与支撑 |

每条目录包含 `rgb.mp4`、`contact_sheet.jpg`、`trajectory.h5`、`model.xml`、`metadata.json` 和 `validation.json`。视频为 640×480、20 FPS，播放速度为仿真原速。图像未后期修改物体运动。

HDF5 保留 T 个原生 12D 动作、控制器目标 13D、T+1 个 RGB / 全状态 / 物体位姿 / 速度 / 末端状态。抽屉任务额外保存逐帧抽屉位置和速度。记录物理接触、阶段、子目标和最终稳定成功判据，完整 MuJoCo 积分状态支持原生动作回放。13D gripper +1=open；native raw gripper +1=close。

采集时只有原生机器人控制动作；抽屉没有驱动器。轨迹内不编辑物体或抽屉 qpos、不绑定、不瞬移、不 reset。脚本使用仿真真值反馈，而非视觉策略。

验证从独立初始化恢复初态，执行保存动作并逐步比对 qpos、qvel、控制目标、物体和末端位姿、抽屉状态、任务谓词和接触标志。另抽取 7 帧重渲染，完整解码 MP4 检查 T+1 帧及 FPS；验证报告记录误差。验证也重算“先开抽屉再举起物体再关上”和“先沿台面推过出口再举起”的物理顺序。最终目标须在末尾连续 1 秒成立。

每个任务目前只有一个成功预览初态，`split=preview_only`，不能据此估计随机场景成功率，也尚不是批量训练集。抽屉是小型台面原型；推抓任务把推动明确写入指令，不声称初态从几何上禁止直接抓取。完整厨房、随机场景、扰动和恢复分支需要后续扩展。

抽屉取罐预览的下降和放置阶段有两次末端位置跟踪超时，误差约 8–11 mm，详情保留在 `metadata.json/phase_timeouts`。实际抓取、直立放置、关抽屉和最终稳定条件均通过；批量扩展前仍需调整抓取高度以减少这两段停留。

采集脚本位于 `/media/tyshao/fail_WM/RoboCasa/complex_interactions_20261022.py`。可在 RoboCasa 根目录执行（采集输出必须使用新目录）：

```bash
.venv/bin/python complex_interactions_20261022.py --task drawer_can --output data/new_drawer_preview
.venv/bin/python complex_interactions_20261022.py --task push_then_bin --output data/new_push_preview
```

复验已有预览：

```bash
.venv/bin/python complex_interactions_20261022.py --task drawer_can --output data/complex_nonblock_20261022/01_drawer_can --validate-only
.venv/bin/python complex_interactions_20261022.py --task push_then_bin --output data/complex_nonblock_20261022/02_push_then_bin --validate-only
```

`summary.json` 汇总实际时长与检查结果；`source_code/` 保留生成代码快照，`checksums.json` 记录交付文件哈希。调试失败的试采保留在 `_calibration/`，不计入成功预览。
