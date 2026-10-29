# 区分真实历史条件误差与自主生成误差

24例，train/val各6失败+6成功。主窗口统一L+2..L+10。每例只有一条beam4输出，不混入采样。
oracle：每一帧都给真实历史再生成下一帧；late：L+2之前给真实历史，此后自由生成；start：只有最初2帧真实图像。

|split|类别|条件|窗口曾抬升|高处帧占比|方块ROI MAE|全帧MAE|真实target NLL|末帧状态|
|---|---|---|---:|---:|---:|---:|---:|---|
|train|failure|oracle|5/6|25.9%|11.92|2.29|0.685|{'table': 5, 'both': 1}|
|train|failure|late|6/6|63.0%|28.74|4.31|1.484|{'high': 5, 'other': 1}|
|train|failure|start|6/6|66.7%|34.01|4.93|1.709|{'high': 4, 'none': 1, 'table': 1}|
|train|success|oracle|6/6|87.0%|9.08|1.50|0.322|{'high': 6}|
|train|success|late|6/6|88.9%|11.80|1.77|0.354|{'high': 6}|
|train|success|start|6/6|88.9%|11.33|1.76|0.361|{'high': 6}|
|val|failure|oracle|6/6|27.8%|13.39|2.62|1.194|{'both': 2, 'table': 4}|
|val|failure|late|6/6|51.9%|23.04|3.78|1.541|{'table': 1, 'high': 3, 'none': 2}|
|val|failure|start|5/6|55.6%|25.98|4.70|1.857|{'none': 1, 'high': 3, 'table': 2}|
|val|success|oracle|6/6|79.6%|15.97|2.35|0.944|{'high': 5, 'none': 1}|
|val|success|late|6/6|79.6%|19.58|2.96|1.083|{'high': 4, 'none': 1, 'table': 1}|
|val|success|start|6/6|81.5%|22.82|3.32|1.147|{'high': 5, 'table': 1}|

注意：成功案例抬升是正确行为，失败案例抬升是错误。高处是红连通块代理，不是检测到了真实夹爪。both/none/other保留。

## 关键差值与区间

按6个匹配初态层bootstrap 10000次，仅探索性诊断。

|比较|类别|条件|指标|均值差|95%区间|
|---|---|---|---|---:|---|
|val_minus_train|failure|oracle|nll|0.5085|[0.4743, 0.5416]|
|autonomous_minus_oracle/train|failure|late|nll|0.7988|[0.3352, 1.3273]|
|autonomous_minus_oracle/train|failure|start|nll|1.0233|[0.6072, 1.4465]|
|autonomous_minus_oracle/val|failure|late|nll|0.3470|[0.2061, 0.5293]|
|autonomous_minus_oracle/val|failure|start|nll|0.6629|[0.4125, 0.9315]|
|val_minus_train|failure|oracle|roi_mae|1.4794|[-1.5929, 5.1689]|
|autonomous_minus_oracle/train|failure|late|roi_mae|16.8261|[11.1720, 23.2972]|
|autonomous_minus_oracle/train|failure|start|roi_mae|22.0920|[14.9461, 29.0292]|
|autonomous_minus_oracle/val|failure|late|roi_mae|9.6429|[6.2147, 11.7884]|
|autonomous_minus_oracle/val|failure|start|roi_mae|12.5855|[8.8032, 15.1388]|
|val_minus_train|failure|oracle|high21_frame_rate|0.0185|[-0.1296, 0.1852]|
|autonomous_minus_oracle/train|failure|late|high21_frame_rate|0.3704|[0.3333, 0.4074]|
|autonomous_minus_oracle/train|failure|start|high21_frame_rate|0.4074|[0.2593, 0.5370]|
|autonomous_minus_oracle/val|failure|late|high21_frame_rate|0.2407|[0.1481, 0.3704]|
|autonomous_minus_oracle/val|failure|start|high21_frame_rate|0.2778|[0.0185, 0.4815]|

## 限制

- 24 cases/12 initial groups/6matched strata; train frames are in-sample, val is previously inspected; no test/generalization population estimate
- Matching factor/phase does not match all physical/image difficulty; gap is evidence consistent with generalization, not proof of overfit
- Oracle vs free varies all generated history including arm/cube; does not isolate cube-only cause
- Original threshold applied to any red connected component; differs from older centroid-of-all-red lift detector, so these counts are not interchangeable with 2x2 report.
- Oracle means GT previous frames with within-frame free beam generation, not target-token teacher forcing. NLL uses separate teacher-forced scoring of that same target.

逐例证据：cases/{train,val}/{id}/comparison.jpg，列为GT/codec/oracle/late/start；每行一个目标帧。完整生成token和浮点图像保留于各mode.npz。
