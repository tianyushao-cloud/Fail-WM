# GT 历史交接实验

进度：153/153 个 模型×案例。抬升阈值 21.2px（GT 未抬起最大 3.1px，GT 抬起最小 39.2px，在验证/所选 split 的 GT 上冻结）。

k=交接帧：k 之前（anchor + 最近 6 帧）用 GT token，之后自回归。标签是相对锚点（抬升阶段首帧 L，或释放阶段首帧 R）的偏移；start 即 k=2，与 E1 相同的开环条件。抬升 = 方块在 [max(k,L), L+10] 内相对 GT 在 L−1 的静止高度上移超过阈值。

## 抬升（未抬起失败应为 not_lifted；成功对照应为 lifted）

|模型|类别|交接|解码|N|抬升|未抬升|不可见|窗口内出现消失|
|---|---|---|---|---:|---:|---:|---:|---:|
|B|success_control|start|beam4|20|20|0|0|1|
|B|success_control|start|sampling|160|159|1|0|85|
|B|success_control|+2|beam4|20|20|0|0|8|
|B|success_control|+2|sampling|160|160|0|0|82|
|B|unlifted_failure|start|beam4|24|12|12|0|11|
|B|unlifted_failure|start|sampling|192|106|86|0|100|
|B|unlifted_failure|+2|beam4|24|12|12|0|10|
|B|unlifted_failure|+2|sampling|192|109|83|0|96|
|C|success_control|start|beam4|20|20|0|0|2|
|C|success_control|start|sampling|160|155|5|0|74|
|C|success_control|+2|beam4|20|20|0|0|3|
|C|success_control|+2|sampling|160|159|1|0|81|
|C|unlifted_failure|start|beam4|24|18|6|0|10|
|C|unlifted_failure|start|sampling|192|135|57|0|124|
|C|unlifted_failure|+2|beam4|24|20|4|0|11|
|C|unlifted_failure|+2|sampling|192|138|54|0|104|
|D|success_control|start|beam4|20|19|1|0|12|
|D|success_control|start|sampling|160|155|5|0|98|
|D|success_control|+2|beam4|20|20|0|0|10|
|D|success_control|+2|sampling|160|157|3|0|93|
|D|unlifted_failure|start|beam4|24|11|13|0|9|
|D|unlifted_failure|start|sampling|192|133|59|0|134|
|D|unlifted_failure|+2|beam4|24|12|12|0|7|
|D|unlifted_failure|+2|sampling|192|108|84|0|104|

## 空中提前释放（释放结束 +3 帧）

|模型|交接|解码|像GT落下|仍在高处|消失|其他|GT缺失|
|---|---|---|---:|---:|---:|---:|---:|
|B|start|beam4|7|0|0|0|0|
|B|start|sampling|32|4|16|4|0|
|B|+2|beam4|6|0|1|0|0|
|B|+2|sampling|43|1|8|4|0|
|C|start|beam4|2|1|3|1|0|
|C|start|sampling|36|2|18|0|0|
|C|+2|beam4|5|0|2|0|0|
|C|+2|sampling|44|4|8|0|0|
|D|start|beam4|5|1|1|0|0|
|D|start|sampling|33|10|13|0|0|
|D|+2|beam4|4|2|1|0|0|
|D|+2|sampling|44|7|5|0|0|

## Codec 底线（GT token 直接解码，同一窗口、同一指标）

```
{
 "success_control": {
  "lifted": 60
 },
 "unlifted_failure": {
  "not_lifted": 69,
  "lifted": 3
 },
 "release_failure": {
  "fell_like_gt": 21
 }
}
```

解读：未抬起失败在 L/+2/+4 交接后仍大量抬升 → 单步动力学乐观（数据/目标问题）；只在 start/−6 抬升 → 自回归漂移。同一案例的 8 次采样不独立，按初态组看 summary.json 的 per_case。逐案例画面：results/<model>/<id>_beam.png（灰框=GT 历史帧）。
