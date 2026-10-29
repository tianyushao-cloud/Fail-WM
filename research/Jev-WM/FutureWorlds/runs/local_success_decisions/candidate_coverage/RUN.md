# 运行与续跑

任务：20个新的成功训练初始场景，4组/8组候选的配对校准。
不训练、不读取验证/测试场景，不调整已固定提示词或评分门槛。
当前报告 REPORT.md 随每个方案评分结束自动更新；summary.json 中 completed_pairs 为完成的配对场景数。
没有结果时的空值不是评分0。

运行日志：run.log
持久化请求预算：attempts.jsonl（包含失败或中断前已登记尝试），全实验上限480。
评分缓存：reviews/，相同输入直接复用，重启不会重新支付已完成评分。
候选缓存：cases/<id>/<four|eight>/candidates.npz，metadata.json包含校验和与实现身份。
评分/选择：同目录 selection.json。没有合格对时选择为空，不制造偏好。
盲化检查：有效候选对对应 blind_0..3.jpg（全部32帧）；blind_mapping.json 为隐藏对应关系。
verification.json：20个不同训练初始状态、排除旧组、原始首两帧、首组4/8候选对齐的实际检查。
runner_snapshot.py：启动时的脚本副本；协议绑定正式 scripts/ 下脚本校验和。
本次不自动启动RL。需要基于完整覆盖率和可视检查结果决定下一步。

若进程停止且实现未改变，可在服务器执行以下命令续跑；flock 防止同时启动两个实验：

    cd /media/tyshao/Jev-WM/FutureWorlds
    nohup flock -n runs/local_success_decisions/candidate_coverage/run.lock       env USE_TF=0 USE_FLAX=0 PYTHONPATH=src:scripts       /media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python -u       scripts/candidate_coverage_decisions.py run       >> runs/local_success_decisions/candidate_coverage/run.log 2>&1 < /dev/null &

API依赖服务器127.0.0.1:17897代理转发及本机Clash；不依赖本次启动用的SSH会话。
密钥只从服务器已有私有文件读取，脚本和报告不保存密钥。
