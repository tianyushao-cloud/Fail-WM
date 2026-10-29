# 对 720 条 E1 结果的事件对齐复核（Opus，2026-10-10）

只读复核，CPU，未改动 evals_e1_expanded_20261009/ 内任何文件。脚本 analyze.py；逐输出 rows.jsonl；汇总 report.json；画面 lift_strip_*.png。

1. 配对 + 初态组 bootstrap（17 失败案例，9 组，sampling）：混合−仅成功 偏目标比例
   全部输出 −11.0pp，95% CI [−22.7, +1.4]；仅可见输出 −3.2pp，CI [−14.6, +8.4]。均不显著。
2. 虚构抬升：GT 未抬起的 7 个失败（grasp_miss 2、contact_without_lift 4、object_off_counter 1），
   GT 方块最大上移 ≤12.3px，GT 抬起案例 ≥38.6px，阈值取中点 25.4px。
   预测上移 >25.4px：仅成功 61/63，混合 59/63（beam4 均 7/7）。lift_strip_*.png 目视确认方块被夹爪带起。
   GT 抬起的成功/失败案例 297/297 均判为抬起（对照）。
3. 空中提前释放（5 例 early_release，释放结束 +3 帧）：
   仅成功 sampling：像 GT 落下 13、仍在高处 13、其他 13、消失 1；beam4：落下 2、高处 1、消失 2。
   混合 sampling：落下 16、高处 4、其他 5、消失 15；beam4：落下 1、消失 4。
   同一批帧的 codec 真实帧重建：方块可见 32/34，中心误差 3–6px。消失主要不是 codec 造成。
4. 混合模型的消失起点集中在 early_release / retreat / lower（释放相关阶段），不是随机缺失。
