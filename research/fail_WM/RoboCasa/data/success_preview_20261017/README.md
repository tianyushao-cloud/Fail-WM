# RoboCasa 成功任务预览（2026-10-17）

三条成功演示，均为真实 MuJoCo 接触仿真，不含人为扰动、轨迹内 reset、物体绑定或瞬移。使用仿真真值反馈的脚本控制，不是视觉策略。

| 目录 | 内容 | 动作时长 |
|---|---|---|
| 01_clear_restore | 将叠在红块上的蓝块移到暂存区；红块放绿色目标；蓝块放回原先 XY 区域的台面 | 37.25 s |
| 02_stack | 将蓝色底块放到目标，再将红块叠在其上并退爪 | 27.25 s |
| 03_sort_pair | 四个方块分到两个绿色标记区域，每区一红一蓝 | 51.95 s |

这批是成功流程效果预览，每个任务只有一个初态，不代表随机场景下的成功率。清障是上下叠放的简化版，尚未验证不存在其他抓取捷径；分拣用台面区域，不是实体容器。

每个目录包含 rgb.mp4（640×480，20 FPS）、contact_sheet.jpg、metadata.json、validation.json、model.xml，以及服务器上的完整 trajectory.h5。预览 zip 不含大型 HDF5。

原始视频没有速度调整、字幕或后期物体运动修改。T 个动作对应 T+1 帧；MP4 比动作时长多 0.05 s。最终成功要求所有物体位置、支撑、释放和速度条件连续满足至少 1 s，并且所有操作子目标均达到。

全部三条已从独立初始化恢复初态，逐步执行保存的原生动作；qpos、qvel、多物体位姿、末端位姿、控制目标最大误差均为 0；事件逐帧一致，最终稳定成功一致。逐条抽查 7 帧重新渲染，最大平均 RGB 误差小于 0.001/255。视频完整解码后帧数及帧率通过。

服务器原始数据：/media/tyshao/fail_WM/RoboCasa/data/success_preview_20261017
脚本：/media/tyshao/fail_WM/RoboCasa/success_preview.py

重新采集（输出目录必须为新目录，防止覆盖）：

```bash
cd /media/tyshao/fail_WM/RoboCasa
.venv/bin/python success_preview.py --task clear_restore --output data/new_success_clear
.venv/bin/python success_preview.py --task stack --output data/new_success_stack
.venv/bin/python success_preview.py --task sort_pair --output data/new_success_sort
```

单独验证已有轨迹：

```bash
.venv/bin/python success_preview.py --task clear_restore --output data/success_preview_20261017/01_clear_restore --validate-only
```

trajectory.h5 的 observations/object_pose_wxyz 为 (T+1,N,7)，object_velocity 为 (T+1,N,6)，object_names 属性确定对象顺序。包含原生 raw12、控制器 target13、多物体状态、RGB、完整积分状态和任务谓词。夹爪符号与旧管线一致：target13 +1=open，native raw[6] +1=close。图像采用分块增量写盘。
