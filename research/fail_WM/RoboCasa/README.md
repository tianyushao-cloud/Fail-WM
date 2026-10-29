# RoboCasa 5 条机器人交互预览数据

服务器目录：`/media/tyshao/fail_WM/RoboCasa`。
这是 RoboCasa 1.0.1 的 `Kitchen` 自定义子任务 `CounterCubePreview`，使用原生 `KitchenArena`、程序化 `Counter` 和物理方块。它不是 RoboCasa365 的标准 benchmark 任务，也未下载完整厨房资产包。机器人为 RoboCasa 支持的 `PandaOmron`：Franka Panda 手臂，底盘和躯干控制指令始终为零（不代表这些自由度在物理中被焊死）。

所有视频直接来自 MuJoCo EGL 渲染；没有改变视频中的物体运动，没有抓取绑定、轨迹内瞬移或 recovery reset。固定第三人称相机，640×480，20 FPS。初始化时设置固定机器人/方块状态并强制刷新控制器缓存，避免 Kitchen reset 后缓存参考位姿滞后。

## 已生成结果

| trajectory | 实际结果 | 动作数 T | RGB 帧 T+1 | 时长 |
|---|---|---:|---:|---:|
| 01_reference | 抓取、举起、松开并放到目标，成功 | 210 | 211 | 10.50 s |
| 02_offset_miss | 抓取点 x 偏移 -0.12 m，抓空，失败 | 213 | 214 | 10.65 s |
| 03_early_close | 下降前提前 0.8 s 闭爪；出现短暂接触，但未举起，失败 | 310 | 311 | 15.50 s |
| 04_early_release | 举起后在空中松爪，掉落在目标之外，失败 | 254 | 255 | 12.70 s |
| 05_recovery | 第一次抓空；无 reset，重新接近实际物体并成功放置 | 317 | 318 | 15.85 s |

成功率 2/5 = 40%。此结果来自仿真状态，不是按扰动分支预设标签。五条共享相同 initial-state ID 和 SHA256，均归 train；只有一个初态组，第一版不构造 val/test。

## 安装与运行

现有服务器环境可以直接运行：

```bash
ssh -p 8081 root@10.20.0.71
cd /media/tyshao/fail_WM/RoboCasa
.venv/bin/python collect.py --output data/new_preview5
.venv/bin/python verify.py --data data/new_preview5
.venv/bin/python statistics.py --data data/new_preview5
```

新 Linux 主机需要 NVIDIA 驱动和可用 EGL/OpenGL；不需要 RTX GPU。下载/解压 `runtime_bundle.tar.gz`（含实际使用的 vendor 源码与模型资产），安装 uv 后：

```bash
bash install.sh
.venv/bin/python collect.py --output data/preview5
.venv/bin/python verify.py --data data/preview5
```

Python 3.11，MuJoCo 3.3.1。`requirements.txt` 为主要运行依赖；`requirements-lock.txt` 记录服务器实际安装版本。`provenance.json` 记录源代码 commit 和资产来源。RoboCasa 需要 robosuite master 的 API，因此使用固定源码快照，而不是仅安装 PyPI robosuite。资产来自 robosuite 1.5.2 wheel；bundle 保留了本次验证的确切组合。没有安装与采集无关的完整训练框架。可通过 `--only reference` 或其他 branch 名单独采集。请给新实验指定新 output，避免覆盖现有预览。

官方参考：[安装](https://robocasa.ai/docs/build/html/introduction/installation.html)、[RoboCasa 源码](https://github.com/robocasa/robocasa)、[robosuite controllers](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/controllers)。

## 数据目录

```text
RoboCasa/
  collect.py, verify.py, statistics.py
  rc_collect/
    environment.py       # 原生厨房/机器人/相机/任务
    initial_state.py     # 固定初态与缓存同步
    controller.py        # 反馈式参考动作与轨迹执行
    perturbations.py     # 扰动分支及参数
    recovery.py          # 依据实际抓取/举起结果决定重试
    events.py            # 实测事件
    action_conversion.py # 13D 转换
    recording.py         # 同步 RGB/state/action 记录
  vendor/{robocasa,robosuite}/
  assets/textures/
  data/preview5/
    summary.json, statistics.json, verification.json, contact_sheet.jpg
    01_reference/ ... 05_recovery/
      rgb.mp4
      trajectory.h5
      metadata.json
      model.xml
```

`trajectory.h5`：

| 字段 | 数量/形状 | 定义 |
|---|---|---|
| actions/raw | T×12 | 实际传入 `env.step` 的 float64 原生动作 |
| actions/target13 | T×13 | 本动作设置后的真实 controller goal，物理积分前采样 |
| actions/timestep, timestamp | T | 动作索引、动作开始时 simulator 时间（秒） |
| actions/phase | T | scripted 执行阶段，仅供追溯，不作结果标签 |
| observations/rgb | (T+1)×480×640×3 | 无损 RGB uint8；MP4 是其有损编码浏览版本 |
| observations/timestep, timestamp | T+1 | 状态/帧索引、simulator 时间（秒） |
| observations/qpos, qvel | T+1 | 全 MuJoCo 广义坐标与速度，包括底盘、躯干、手臂、手指、方块 |
| observations/robot_joint_position, robot_joint_velocity | (T+1)×7 | Panda 手臂关节角 rad / 速度 rad/s |
| observations/eef_executed13 | (T+1)×13 | 实际 grip_site 世界位姿及实际夹爪开度 |
| observations/gripper_joint_position | (T+1)×2 | 两根手指的真实关节位置 m |
| observations/actuator_ctrl | T+1 | 真实底层 actuator controls |
| observations/object_pose_wxyz | (T+1)×7 | 世界 xyz（m）+ 四元数 wxyz |
| observations/object_velocity | (T+1)×6 | 方块线速度与角速度 |
| events/flags | (T+1)×7 | contact, grasp, lift, drop, release, place, success |
| events/gripper_cube_normal_force | T+1 | 夹爪–方块接触法向力之和 N |
| simulator/state | T+1 | robosuite 可回放 flattened state |
| simulator/integration_state | T+1 | MuJoCo 全积分状态，包含 warmstart/ctrl 等 |
| simulator/initial_gripper_action | 2 | 回放时恢复夹爪内部积分目标 |

`metadata.json` 包括 episode/initial-state ID、初态 hash、seed、split、实际最终 success/failure_type、事件起始时刻、每次 grasp-and-lift 尝试的实测结果、扰动参数和阶段索引。每个 `actions/phase` 及 simulator timestamp 可精确定位扰动。失败恢复不 reset，第二次抓取使用当时的实际方块位置。

## 原生动作与 13D 定义

原生 12D 控制接口没有被改成 13D：

```text
raw[0:3]   = OSC_POSE 世界 xyz 增量（归一化，范围 [-1,1]，映射 ±0.04 m）
raw[3:6]   = 世界轴角旋转增量（归一化，映射 ±0.25 rad/分量）
raw[6]     = gripper 方向指令，+1 闭合，-1 打开，0 保持
raw[7:10]  = base 控制，本实验为 0
raw[10]    = torso 控制，本实验为 0
raw[11]    = hybrid mode，本实验为 -1（arm mode）
```

`actions/target13[i] = [x,y,z,r11,r12,r13,r21,r22,r23,r31,r32,r33,g]`：世界坐标位置（m）+ grip_site 旋转矩阵（行优先，列向量是局部坐标轴在世界中的方向）+ 夹爪内部积分后的归一化 actuator 目标，**g=+1 打开，g=-1 闭合**。这与 native gripper 方向指令的符号不同，且不是方块是否被抓住的标签。

`observations/eef_executed13[i]` 则来自实际机器人状态，夹爪部分为 `clip(2 * (abs(q_left)+abs(q_right)) / 0.08 - 1, -1, 1)`。实际 pose 不是 commanded action，不能把下一时刻 executed pose 当作无泄漏的预测动作输入。

转换实现位于 `rc_collect/action_conversion.py`。记录器在 `_pre_action(..., policy_step=True)` 调用原生控制器设置 goal 之后、物理积分之前，读取 `goal_pos`, `goal_ori`, `gripper.current_action`。因此保存的是实际 controller target，未用下一帧的执行结果代替目标。

用于 world model 的建议配对：

```text
RGB[i], state[i], target13[i]  ->  RGB[i+1], state[i+1]
```

原生动作也可作为独立输入。T 个动作对应 T 次 transition，而 RGB/state 保留 T+1 个采样点；最后一帧没有后续动作。时间步严格为 0.05 s，不使用墙钟或渲染耗时作为仿真时间。MP4 从初始帧开始，因此比 T/20 秒多一个帧周期。

## 事件与验证

contact：真实夹爪–方块碰撞；grasp：robosuite 两侧 fingerpad 接触检查（可能是瞬时接触，不保证稳定抓住）；lift：方块中心高于台面静止中心 0.06 m；drop：此前已 lift、当前无 grasp、仍在空中且实际向下速度 < -0.15 m/s；release：grasp 丢失且实际夹爪开度增大；place：方块与台面接触且 XY 距目标 < 0.045 m；success：还需高度正确、方块速度 < 0.06 m/s、末端远离方块 > 0.10 m。所有标签读取 simulator 实测状态。

`verify.py` 已检查五条：T+1 RGB/state/MP4 帧数量、动作起始 timestamp、严格 0.05 s 步长、rotation matrix 正交性；从保存初态执行原生 action 的 open-loop replay，逐步 qpos 最大误差 **0**，最终结果一致。每条抽样 7 帧直接重渲染，与无损 RGB 的单通道最大差 **1/255**（GPU 渲染舍入），不是逐帧全量图像比对。报告见 `verification.json`。

## 当前限制与扩展

这是验证格式/物理差异的五条共享初态样本，不代表数据分布或成功率估计。场景为最小自定义 RoboCasa 厨房台面，没有完整多厨房 benchmark 资产；视觉为 MuJoCo OpenGL，不是 Isaac Sim RTX/PBR。目标位移约 10 cm，方块 4 cm；后续可改变工作区或相机提高视觉差异。

事件在 20 Hz 的状态采样点标注，可能漏掉 <50 ms 的接触；drop/release/lift 的阈值是本任务尺度下的启发式。短暂双指接触单独计 grasp，稳定抓取通过 grasp-and-lift 联合判断。脚本使用 simulator 的真实方块位置反馈，而不是视觉识别。

扩展时先下载官方完整 assets，替换 environment/reference controller 为 CounterToSink、CounterToCabinet 等官方任务；保留同样 recording/action conversion/replay 结构。用 task-specific 目标、接触、支撑与终止条件替换 success/place 标签。采样多个初态时先按 initial-state group 分 train/val/test，再生成各组动作分支；不能把同一初态的成功和失败分到不同 split。

## v2.1 采样同步修正

更细的检查发现，MuJoCo `mj_step2` 更新 qpos 后，派生的末端/物体变换与渲染缓存可能仍是前一 2 ms 物理子步。仅检查 T/T+1、时间戳和相同代码的回放不能发现这种一致的缓存滞后。现在采样前执行 `mj_forward` 刷新当前 qpos 的几何、接触和渲染信息，并恢复完整积分状态（包括 solver warmstart），不推进时间、不改变后续物理积分。

已经录制的原生动作通过真实 open-loop replay 重新录制 RGB/derived state，逐步验证原始 qpos、qvel 与 target13 不变，raw action 字节一致；不是对视频中的运动做后期修改。旧缓存版本仅保留在 `_cache2ms_backup` 审计目录，不应作为额外训练数据。此前的五条预览也更新为此版本。

新增验证逐步核对 executed13 与 object pose 是否等于当前 qpos 的真实几何结果，并独立按原生 OSC 增量和夹爪积分公式核对 target13。采样逻辑在 `rc_collect/synchronization.py`，修正工具为 `repair_sampling_sync.py`，独立转换核对为 `audit_action_conversion.py`。
