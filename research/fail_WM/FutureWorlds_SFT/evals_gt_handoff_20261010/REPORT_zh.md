# GT 历史交接实验

进度：102/102 个 模型×案例。抬升阈值 21.2px（GT 未抬起最大 3.1px，GT 抬起最小 39.2px，在验证/所选 split 的 GT 上冻结）。

k=交接帧：k 之前（anchor + 最近 6 帧）用 GT token，之后自回归。标签是相对锚点（抬升阶段首帧 L，或释放阶段首帧 R）的偏移；start 即 k=2，与 E1 相同的开环条件。抬升 = 方块在 [max(k,L), L+10] 内相对 GT 在 L−1 的静止高度上移超过阈值。

## 抬升（未抬起失败应为 not_lifted；成功对照应为 lifted）

|模型|类别|交接|解码|N|抬升|未抬升|不可见|窗口内出现消失|
|---|---|---|---|---:|---:|---:|---:|---:|
|natural|success_control|start|beam4|20|20|0|0|2|
|natural|success_control|start|sampling|160|160|0|0|58|
|natural|success_control|-6|beam4|20|20|0|0|0|
|natural|success_control|-6|sampling|160|160|0|0|31|
|natural|success_control|-3|beam4|20|20|0|0|0|
|natural|success_control|-3|sampling|160|160|0|0|31|
|natural|success_control|+0|beam4|20|20|0|0|1|
|natural|success_control|+0|sampling|160|160|0|0|43|
|natural|success_control|+2|beam4|20|20|0|0|2|
|natural|success_control|+2|sampling|160|160|0|0|33|
|natural|success_control|+4|beam4|20|20|0|0|2|
|natural|success_control|+4|sampling|160|160|0|0|30|
|natural|unlifted_failure|start|beam4|24|19|5|0|10|
|natural|unlifted_failure|start|sampling|192|151|41|0|111|
|natural|unlifted_failure|-6|beam4|24|17|7|0|6|
|natural|unlifted_failure|-6|sampling|192|152|40|0|93|
|natural|unlifted_failure|-3|beam4|24|20|4|0|13|
|natural|unlifted_failure|-3|sampling|192|144|48|0|105|
|natural|unlifted_failure|+0|beam4|24|18|6|0|8|
|natural|unlifted_failure|+0|sampling|192|151|41|0|94|
|natural|unlifted_failure|+2|beam4|24|16|8|0|5|
|natural|unlifted_failure|+2|sampling|192|148|44|0|87|
|natural|unlifted_failure|+4|beam4|24|17|7|0|7|
|natural|unlifted_failure|+4|sampling|192|142|50|0|81|
|success|success_control|start|beam4|20|20|0|0|0|
|success|success_control|start|sampling|160|160|0|0|10|
|success|success_control|-6|beam4|20|20|0|0|0|
|success|success_control|-6|sampling|160|160|0|0|2|
|success|success_control|-3|beam4|20|20|0|0|0|
|success|success_control|-3|sampling|160|160|0|0|6|
|success|success_control|+0|beam4|20|20|0|0|0|
|success|success_control|+0|sampling|160|160|0|0|5|
|success|success_control|+2|beam4|20|20|0|0|0|
|success|success_control|+2|sampling|160|160|0|0|9|
|success|success_control|+4|beam4|20|20|0|0|0|
|success|success_control|+4|sampling|160|160|0|0|7|
|success|unlifted_failure|start|beam4|24|24|0|0|7|
|success|unlifted_failure|start|sampling|192|187|5|0|74|
|success|unlifted_failure|-6|beam4|24|23|1|0|7|
|success|unlifted_failure|-6|sampling|192|183|9|0|68|
|success|unlifted_failure|-3|beam4|24|23|1|0|7|
|success|unlifted_failure|-3|sampling|192|189|3|0|56|
|success|unlifted_failure|+0|beam4|24|23|1|0|3|
|success|unlifted_failure|+0|sampling|192|186|6|0|58|
|success|unlifted_failure|+2|beam4|24|23|1|0|3|
|success|unlifted_failure|+2|sampling|192|187|5|0|56|
|success|unlifted_failure|+4|beam4|24|24|0|0|8|
|success|unlifted_failure|+4|sampling|192|187|5|0|59|

## 空中提前释放（释放结束 +3 帧）

|模型|交接|解码|像GT落下|仍在高处|消失|其他|GT缺失|
|---|---|---|---:|---:|---:|---:|---:|
|natural|start|beam4|1|1|5|0|0|
|natural|start|sampling|15|8|27|6|0|
|natural|-6|beam4|3|0|4|0|0|
|natural|-6|sampling|19|8|26|3|0|
|natural|-3|beam4|3|2|2|0|0|
|natural|-3|sampling|21|12|21|2|0|
|natural|+0|beam4|2|0|5|0|0|
|natural|+0|sampling|19|6|30|1|0|
|natural|+2|beam4|2|0|5|0|0|
|natural|+2|sampling|18|7|29|2|0|
|success|start|beam4|3|2|0|2|0|
|success|start|sampling|16|22|6|12|0|
|success|-6|beam4|3|2|0|2|0|
|success|-6|sampling|27|16|5|8|0|
|success|-3|beam4|5|0|0|2|0|
|success|-3|sampling|29|12|10|5|0|
|success|+0|beam4|3|1|1|2|0|
|success|+0|sampling|13|16|15|12|0|
|success|+2|beam4|2|1|3|1|0|
|success|+2|sampling|22|17|11|6|0|

## Codec 底线（GT token 直接解码，同一窗口、同一指标）

```
{
 "success_control": {
  "lifted": 40
 },
 "unlifted_failure": {
  "not_lifted": 36,
  "lifted": 12
 },
 "release_failure": {
  "fell_like_gt": 8,
  "vanished": 6
 }
}
```

解读：未抬起失败在 L/+2/+4 交接后仍大量抬升 → 单步动力学乐观（数据/目标问题）；只在 start/−6 抬升 → 自回归漂移。同一案例的 8 次采样不独立，按初态组看 summary.json 的 per_case。逐案例画面：results/<model>/<id>_beam.png（灰框=GT 历史帧）。
