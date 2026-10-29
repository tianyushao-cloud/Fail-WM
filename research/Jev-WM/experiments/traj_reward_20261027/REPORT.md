# 纯轨迹 reward 离线评测 (2026-10-27)

数据：v0 8-GPU state-reward run 的 judge_log，480 组 × 8 候选 × 32 帧（256×320）。全部 CPU 计算（compute.py，48 进程约 4 分钟）。

## 方法 (traj.py, cam.py)
- 相机：preview 相机解析投影（pos .9 -1.4 1.9, fovy 48, 640×480）× 缩放到 320×256；用 GT 帧 cube tracker 验证 rms 0.3/0.6 px。h5 里的 cube/eef 3D 位置 → GT 2D 轨迹。
- cube 项：GT 红色 mask 与预测红色 mask 的对称平均距离（所有红色像素，不只取最大 blob → ghost cube 计入）；GT 被遮挡时用投影的 cube 中心；预测完全没有红色 = 固定 8 px（第一版按上限 40 计，误伤被手臂遮挡的正确预测）。每帧上限 80。
- 夹爪项：GT 帧夹爪 patch（投影 eef 处 48×48）在预测帧 ±40 px 内做模板匹配 = 预测臂相对 GT 的位移。GT-vs-GT 验证（val_grip.py）：位移≤20 px 时中位误差 0.5–1.9 px，30 px 以上开始出现尾部错误。
- 相对项：(预测 cube − 预测夹爪) vs (GT cube − GT 夹爪)。
- 聚合：每帧代价 → 0.5·均值 + 0.5·最差 1/4 均值（与 rule 相同），reward = 1 − agg/cap。

## 结果 (report2.txt, report3.txt, fidelity.txt)
排序准确率，标签 = tracker 抬升时间线 vs h5 lift（基准定义；只看抬没抬起）：

| | rule | VLM state v0 | v0 训练总 | T_cube | T_full(.5/.25/.25) | T_cube+VLM |
|---|---|---|---|---|---|---|
| ALL (447 对) | .79 | .87 | .89 | .84 | .83 | .92 |
| grasp_miss | .88 | .97 | .99 | 1.00 | .99 | .99 |
| contact_without_lift | .58 | .76 | .69 | .74 | .74 | .82 |
| object_off_counter | .80 | .84 | .91 | .64 | .64 | .90 |
| 全 fake 组 (34 组) | .60 | .82 | .87 | .89 | .91 | .91 |
| 伪影对（lift 一致） | .95 | .63 | .90 | .78 | .68 | .70 |
| 组内 Spearman vs −ROI MSE | +.33 | **−.04** | +.26 | +.35 | **+.51** | +.14 |

- 夹爪项单独排 lift 标签 ≈ 随机（.49），但与像素保真度相关 +.71。VLM state 与像素保真度无关（−.04），这和 v0 RL 后 ROI LPIPS 变差一致。
- 盲评（blind.py，T 与标签冲突的 10 对，contact_without_lift / object_off_counter）：我判断与 T 一致 5，与标签一致 3，平 2 → 这两类上 T 的低分很大一部分是标签问题。
- 已知弱点：2D 视角下"夹在爪里"与"在爪正下方的桌上"只差 ~10 px（blind_00）；object_off_counter 后半段 WM 的臂整体不跟动作（夹爪位移 22.7 px），模板匹配搜索半径 40 在这里会饱和。
- 加入 rule 的 persistence 项（.2/.3）会把全 fake 组排序打到 .66/.54 → 不要加：它重罚"消失"，等于奖励把 cube 一直带着。
