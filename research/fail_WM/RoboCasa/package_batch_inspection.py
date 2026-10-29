"""Package generated preview copies and audit reports; full H5 stays on server."""
from pathlib import Path
import json, zipfile, shutil

project = Path(__file__).resolve().parent
root = project / "data/batch1000"
stats = json.loads((root / "dataset_statistics.json").read_text())
audit = json.loads((root / "final_audit.json").read_text())
assert stats["episodes"] == 1000 and audit["all_pass"]
analysis = root / "analysis_tools"
analysis.mkdir(exist_ok=True)
for name in ["audit_dataset.py", "audit_action_conversion.py", "batch_report.py"]:
    shutil.copy2(project / name, analysis / name)

text = f"""# RoboCasa 1000 条四进程采集结果

完整数据：`{root}`，1000 条，约 {stats['episode_bytes']/1e9:.2f} GB。
本压缩包是原始仿真视频抽检集，不包含全部 HDF5。解压后打开 `inspection/index.html`。

实测最终结果：555 成功、445 失败；抓空 170、接触未抬起 116、掉落 56、抬起未放到目标 81、离开台面 22。
300 参考执行 / 500 单因素扰动 / 100 双因素 / 100 恢复。恢复成功 93、失败 7。
100 个初态组分为 80/10/10 组，train/val/test 为 800/100/100 条，组间无泄漏。

全量原生动作回放、MP4 完整解码、T/T+1 数量和时间戳检查通过。
原生动作是 12D；target13 是世界坐标 xyz + 行优先 3x3 旋转矩阵 + 归一化夹爪目标（+1 打开、-1 闭合）。
executed13 是实际执行位姿和开度，另存于 observations，不能当作预测下一帧的 commanded action。
训练配对：RGB[i], state[i], target13[i] -> RGB[i+1], state[i+1]；20 Hz，T actions 对应 T+1 frames。

MuJoCo 的 2 ms 派生几何缓存滞后已修正：采样前刷新当前 qpos 的几何，并恢复完整积分状态。
已录制动作真实回放重新采样，无损保留 raw action、qpos、qvel、target13；不是后期改视频。
旧 `_cache2ms_backup` 目录只用于审计，不属于这 1000 条数据。
569 条原生动作来自 v2.0 控制反馈，431 条来自 v2.1 同步反馈；全部观测采用 v2.1 同步采样，metadata 可追溯。
旧五条 RoboCasa 分享包也已更新，应使用修正版 ZIP。

当前是 RoboCasa Kitchen 自定义简单台面任务、Franka Panda 手臂、MuJoCo EGL/OpenGL（640x480），非完整 RoboCasa365/RTX。
事件是 20 Hz 实测启发式标签，可能漏掉短暂接触；success 事件表示曾达到条件，最终 success 是结束时状态，所以两者数量可能不同。
代码、安装/运行、字段和定义见 README.md、BATCH1000_README.md、pipeline_code.zip。
"""
(root / "RESULTS_README.md").write_text(text)
codefiles = [p for p in project.glob("*.py") if p.name != "resume_after_sync.py"]
codefiles += list((project / "rc_collect").glob("*.py"))
codefiles += [project / n for n in ["README.md", "BATCH1000_README.md", "requirements.txt", "requirements-lock.txt", "install.sh", "provenance.json"]]
with zipfile.ZipFile(project / "pipeline_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in codefiles:
        z.write(p, p.relative_to(project))

target = project / "RoboCasa_batch1000_inspection.zip"
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted((root / "inspection").rglob("*")):
        if p.is_file(): z.write(p, "RoboCasa_batch1000/" + str(p.relative_to(root)))
    for name in ["dataset_statistics.json", "final_audit.json", "action_conversion_audit.json", "RESULTS_README.md", "environment_provenance.json", "splits.json", "plan.json"]:
        z.write(root / name, "RoboCasa_batch1000/" + name)
    for name in ["README.md", "BATCH1000_README.md", "pipeline_code.zip"]:
        z.write(project / name, "RoboCasa_batch1000/" + name)
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert len([n for n in z.namelist() if n.endswith(".mp4")]) == 15
print(json.dumps({"path":str(target),"bytes":target.stat().st_size,"videos":15}))
