# 20场景：4候选与8候选对照

状态：COMPLETE，完成 20/20 个配对场景；API尝试 309/480。RL更新0。

提示词、组件评分与可靠对门槛固定；原始首两帧条件、32帧预测、成功训练数据。

|方案|已评分场景|有可靠对场景|有效率|平均不同轨迹数|
|---|---:|---:|---:|---:|
|four|20|5|0.25|3.85|
|eight|20|12|0.6|7.45|

配对结果：
{"rate_difference_eight_minus_four": 0.35, "bootstrap_95_percent_ci": [0.15, 0.55], "both": 5, "four_only": 0, "eight_only": 7, "neither": 8}

这些是候选覆盖率，不是RL涨点。多候选也增加误选机会，必须结合盲化图检查语义正确性。
LPIPS只作离线诊断，不进入奖励。不得只因候选与LPIPS一致，就认定物体行为正确。
API回执见 reviews/；逐例指标和选择见 cases/<id>/<four|eight>/；blind_0..3.jpg 展示全部32帧，blind_mapping.json 为对应关系。
本实验只比较候选数，groups=4/8，beams=2，penalty=0.1，其余不变。没有自动启动训练。
