# GT 历史交接实验

进度：102/102 个 模型×案例。抬升阈值 21.2px（GT 未抬起最大 3.1px，GT 抬起最小 39.2px，在验证/所选 split 的 GT 上冻结）。

k=交接帧：k 之前（anchor + 最近 6 帧）用 GT token，之后自回归。标签是相对锚点（抬升阶段首帧 L，或释放阶段首帧 R）的偏移；start 即 k=2，与 E1 相同的开环条件。抬升 = 方块在 [max(k,L), L+10] 内相对 GT 在 L−1 的静止高度上移超过阈值。

## 抬升（未抬起失败应为 not_lifted；成功对照应为 lifted）

|模型|类别|交接|解码|N|抬升|未抬升|不可见|窗口内出现消失|
|---|---|---|---|---:|---:|---:|---:|---:|
|c34_natural|success_control|start|beam4|20|20|0|0|4|
|c34_natural|success_control|start|sampling|160|158|2|0|74|
|c34_natural|success_control|+2|beam4|20|20|0|0|8|
|c34_natural|success_control|+2|sampling|160|160|0|0|74|
|c34_natural|unlifted_failure|start|beam4|24|12|12|0|5|
|c34_natural|unlifted_failure|start|sampling|192|107|85|0|94|
|c34_natural|unlifted_failure|+2|beam4|24|14|10|0|4|
|c34_natural|unlifted_failure|+2|sampling|192|117|75|0|88|
|c34_success|success_control|start|beam4|20|20|0|0|0|
|c34_success|success_control|start|sampling|160|160|0|0|9|
|c34_success|success_control|+2|beam4|20|20|0|0|0|
|c34_success|success_control|+2|sampling|160|160|0|0|22|
|c34_success|unlifted_failure|start|beam4|24|23|1|0|11|
|c34_success|unlifted_failure|start|sampling|192|180|12|0|111|
|c34_success|unlifted_failure|+2|beam4|24|23|1|0|9|
|c34_success|unlifted_failure|+2|sampling|192|189|3|0|87|

## 空中提前释放（释放结束 +3 帧）

|模型|交接|解码|像GT落下|仍在高处|消失|其他|GT缺失|
|---|---|---|---:|---:|---:|---:|---:|
|c34_natural|start|beam4|3|0|4|0|0|
|c34_natural|start|sampling|27|7|19|3|0|
|c34_natural|+2|beam4|3|0|4|0|0|
|c34_natural|+2|sampling|33|1|20|2|0|
|c34_success|start|beam4|2|0|1|4|0|
|c34_success|start|sampling|20|9|11|16|0|
|c34_success|+2|beam4|3|3|0|1|0|
|c34_success|+2|sampling|21|10|13|12|0|

## Codec 底线（GT token 直接解码，同一窗口、同一指标）

```
{
 "success_control": {
  "lifted": 40
 },
 "unlifted_failure": {
  "not_lifted": 46,
  "lifted": 2
 },
 "release_failure": {
  "fell_like_gt": 14
 }
}
```

解读：未抬起失败在 L/+2/+4 交接后仍大量抬升 → 单步动力学乐观（数据/目标问题）；只在 start/−6 抬升 → 自回归漂移。同一案例的 8 次采样不独立，按初态组看 summary.json 的 per_case。逐案例画面：results/<model>/<id>_beam.png（灰框=GT 历史帧）。
