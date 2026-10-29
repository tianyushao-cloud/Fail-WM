# RoboCasa 四进程 1000 条 action→outcome 数据

服务器：`/media/tyshao/fail_WM/RoboCasa/data/batch1000`。
本次仍使用已验证的 RoboCasa Kitchen 自定义最小台面任务 `CounterCubePreview`、PandaOmron 的 Franka Panda 手臂、MuJoCo EGL/OpenGL 固定第三人称相机（640×480 / 20 Hz）。不是完整 RoboCasa365 benchmark 或 RTX 渲染。

## 采样与划分

100 个初态组，每组 10 条动作分支：3 条参考控制风格、5 条单因素扰动、1 条双因素组合、1 条恢复分支。总计划 300 / 500 / 100 / 100。参考分支采用不同接近高度、动作持有时长和安全搬运路径；不会重复复制同一视频来凑数量。

每组采样方块 XY、方块 yaw 和目标 XY，目标与初始方块相距 12–22 cm。物体尺寸固定 4 cm。模型位置和状态仅在录制前设置；轨迹内不瞬移、不绑定物体、不 reset。物理扰动分支改变质量/摩擦，保留相同父初态组用于保守划分。

全量初态组在采集前一次性分为 train 80 组 / val 10 组 / test 10 组。因此相应 episode 为 800 / 100 / 100；同一初态组所有动作分支只能出现在同一 split。物理参数不同的分支也不会被拆到其他 split。

单因素覆盖：抓取位置偏移、末端倾斜、提前闭爪、举起后延迟闭爪、不完全闭爪、搬运中提前松爪、放置目标偏移、搬运距离不足、暂停并跳过下降、推偏后使用旧抓取点、修改中间 waypoint、质量、摩擦。幅度由 action_seed 采样，小扰动可以成功，所有结局由 simulator 判定。

恢复分支分别尝试抓空/提前闭爪/掉落/推偏后的交互。只在实际未 grasp-and-lift 时重试；第二次接近使用当前物体位置和 yaw，不 reset。若物体已落到台面之外，会继续接近可达边缘并观察，记录恢复失败，不把物体放回去。

## 执行与续采

```bash
cd /media/tyshao/fail_WM/RoboCasa
.venv/bin/python batch_collect.py --output data/batch1000 --episodes 1000 --workers 4
# 四进程小批次验证：
.venv/bin/python batch_collect.py --output data/batch1000 --episodes 1000 --workers 4 --limit 40
# 完成小批次后运行第一条命令，自动保留并跳过已验证的 40 条。
```

四个独立 spawn 进程，各自拥有 MuJoCo / EGL 环境及 episode 文件；每个进程完成 25 条后轮换，避免长期渲染资源积累。OMP/BLAS/Numba 与视频编码线程设为 1。episode 在 `incomplete/` 完成采集和验证后，再原子移动到 `episodes/`，只有完整、验证通过的 episode 计入完成数量。采集/回放错误最多尝试 3 次并记录 `errors/`；任务物理失败正常保存，不进行成功筛选。重试不会为了获得成功而改变 action_seed。

## 文件和对齐

```text
batch1000/
  plan.json                  # 所有 episode 初态、split、扰动意图，计划 hash
  progress.json              # 随采集更新的实际完成/成功/失败/验证统计
  statistics.json            # 本次运行结束后的统计
  manifest.jsonl, splits.json
  source_code/               # 采集代码快照
  episodes/g00000_b00/
    rgb.mp4
    trajectory.h5
    metadata.json
    model.xml
    validation.json
  errors/                    # 执行异常记录，物理失败不属于 errors
  incomplete/                # 未完成/重试中的 episode
```

每条 T 个 `actions/raw`（原生 12D），T 个 `actions/target13`（控制器真实目标），T+1 个 RGB/state。`observations/eef_executed13` 是实际物理结果，不能替代预测输入动作。格式沿用五条预览版，详见主 README.md。

13D：世界 xyz + 行优先旋转矩阵 9D + gripper 1D。target gripper +1=open，-1=closed；native raw[6] 为方向指令，+1=close，-1=open。raw[7:11] 为零底盘/躯干指令，raw[11]=-1 arm mode。原生接口保持 12D，未改成 13D。

`actual_phase_intervals` 逐段记录阶段、动作索引和实际 simulator 起止时间，包含 attempt0/attempt1，用于定位扰动和恢复。`attempt_results`、`events` 来自实际 contact/grasp/lift/drop/release/place/success。最终 failure_type 依据实际事件与物体位置，包含 grasp_miss、contact_without_lift、dropped_before_goal、lifted_but_not_placed、object_off_counter。

## 每条验证

逐条检查原生 action 维度、T/T+1 长度、索引、0.05 s 时间步、动作起始时间与当前状态时间一致、数值有限、旋转矩阵正交/行列式为 1；完整解码 MP4 验证帧数、FPS 和时长。MP4 有初始帧，容器时长为 (T+1)/20 秒，transition 时长为 T/20 秒。

每条执行原生动作的 open-loop replay：重新 reset 控制器并恢复完整初始积分状态和夹爪内部目标，逐步比对 qpos/qvel 和控制器 target13，抽样 7 帧重渲染，核对最终 success。通过阈值 qpos<1e-6、qvel<1e-5、target13<1e-6；重渲染 RGB 平均误差 <0.05/255。RGB 抽样允许 GPU 渲染舍入，不要求每个像素 bitwise 相同。

## 实际覆盖与限制

300/500/100/100 是采样意图配比，不是成功/失败配比；最终统计以 `statistics.json` 为准。物理失败、恢复失败都保留。接触等事件在 20 Hz 采样点检查，可能漏掉短暂事件；双指 contact 可能只是瞬时接触，稳定抓取还需观察 lift。参考控制使用 simulator 对象 pose 反馈，不是视觉控制。

不同初态依然属于同一个简单台面任务，不等于跨厨房、多物体或多任务泛化。训练应使用 initial-state split，统一 world-frame action13，并对 gripper/state 的定义保持一致。比较成功-only 与 success+failure+recovery 时，需要匹配训练 transition 数量和 held-out 初态组。

## v2.1 采样同步修正

更细的检查发现，MuJoCo `mj_step2` 更新 qpos 后，派生的末端/物体变换与渲染缓存可能仍是前一 2 ms 物理子步。仅检查 T/T+1、时间戳和相同代码的回放不能发现这种一致的缓存滞后。现在采样前执行 `mj_forward` 刷新当前 qpos 的几何、接触和渲染信息，并恢复完整积分状态（包括 solver warmstart），不推进时间、不改变后续物理积分。

已经录制的原生动作通过真实 open-loop replay 重新录制 RGB/derived state，逐步验证原始 qpos、qvel 与 target13 不变，raw action 字节一致；不是对视频中的运动做后期修改。旧缓存版本仅保留在 `_cache2ms_backup` 审计目录，不应作为额外训练数据。此前的五条预览也更新为此版本。

新增验证逐步核对 executed13 与 object pose 是否等于当前 qpos 的真实几何结果，并独立按原生 OSC 增量和夹爪积分公式核对 target13。采样逻辑在 `rc_collect/synchronization.py`，修正工具为 `repair_sampling_sync.py`，独立转换核对为 `audit_action_conversion.py`。
