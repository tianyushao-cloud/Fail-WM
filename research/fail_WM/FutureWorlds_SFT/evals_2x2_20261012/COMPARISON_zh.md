# codec34k 2×2 数值对照

原始结果：comparison.json；以初态组bootstrap，10000次。
缺失单列，不把不可见解释成正确预测失败。LPIPS@20是前20个生成帧的平均。

|组|条件|关键失败虚构抬升|不可判|成功抬升|提前释放消失|
|---|---|---:|---:|---:|---:|
|A|start|103/135 (76.30%)|0|98.89%|36.51%|
|A|+2|120/135 (88.89%)|0|100.00%|38.10%|
|B|start|106/135 (78.52%)|0|99.44%|25.40%|
|B|+2|104/135 (77.04%)|0|100.00%|14.29%|
|C|start|112/135 (82.96%)|0|97.22%|33.33%|
|C|+2|114/135 (84.44%)|0|99.44%|15.87%|
|D|start|103/135 (76.30%)|0|96.67%|22.22%|
|D|+2|96/135 (71.11%)|0|98.33%|9.52%|

逐项护栏与分组置信区间见comparison.json。

{
  "B": {
    "checks": {
      "success_ge_95pct/start": true,
      "partial_close_le_A/start": true,
      "release_vanished_le_A/start": true,
      "success_ge_95pct/+2": true,
      "partial_close_le_A/+2": false,
      "release_vanished_le_A/+2": true,
      "lpips_mean_first20_le_1.05A": false
    },
    "all_pass": false
  },
  "C": {
    "checks": {
      "success_ge_95pct/start": true,
      "partial_close_le_A/start": false,
      "release_vanished_le_A/start": true,
      "success_ge_95pct/+2": true,
      "partial_close_le_A/+2": false,
      "release_vanished_le_A/+2": true,
      "lpips_mean_first20_le_1.05A": false
    },
    "all_pass": false
  },
  "D": {
    "checks": {
      "success_ge_95pct/start": true,
      "partial_close_le_A/start": false,
      "release_vanished_le_A/start": true,
      "success_ge_95pct/+2": true,
      "partial_close_le_A/+2": false,
      "release_vanished_le_A/+2": true,
      "lpips_mean_first20_le_1.05A": false
    },
    "all_pass": false
  }
}
