"""Prepare a four-GPU launch; preserve prior world-size artifacts and stop only this task."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    import torch
    visible = torch.cuda.device_count()
    print(f"Visible GPUs: {visible}; required for launch: 4", flush=True)
    if args.inspect:
        return
    if visible < 4:
        raise RuntimeError("Need four GPUs visible in this environment. Existing one-GPU job was left running.")
    from runtime import api_environment, wait_for_api
    api_environment()
    wait_for_api()
    print("Private credential and complete Decisions HTTPS route verified", flush=True)
    judge_config = json.loads((ROOT / "train.json").read_text())["openai_judge"]
    print("Scoring concurrency: %d workers/GPU, %d global; timeout %ds" %
          (judge_config["workers"], 4 * judge_config["workers"], judge_config["timeout_seconds"]), flush=True)
    lock = (ROOT / ".launcher.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        status = json.loads((ROOT / "status.json").read_text())
        pid = int(status["pid"])
        proc = Path(f"/proc/{pid}")
        command = (proc / "cmdline").read_bytes().decode().strip("\0").split("\0")
        if not any(Path(arg).name == "run.py" for arg in command):
            raise RuntimeError("Lock holder is not the expected launcher; no process stopped")
        if (proc / "cwd").resolve() != ROOT:
            raise RuntimeError("Launcher belongs to another directory; no process stopped")
        if "--gpus" not in command or command[command.index("--gpus") + 1] != "1":
            raise RuntimeError("A different GPU configuration is already running; no process stopped")
        if os.getpgid(pid) != pid:
            raise RuntimeError("One-GPU launcher has no isolated process group; stop it manually first")
        print(f"Stopping this experiment's one-GPU process group {pid}", flush=True)
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 20
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Previous launcher has not released its lock")
                time.sleep(.2)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = []
    changed_formal_world = False
    for name, state in [("pilot", ROOT / "pilot/train/run.json"), ("train", ROOT / "train/run.json")]:
        if not state.exists():
            continue
        old_world = json.loads(state.read_text())["config"]["world_size"]
        if old_world == 4:
            continue
        folder = ROOT / name
        target = folder.with_name(f"{name}.world{old_world}.{stamp}")
        folder.rename(target)
        archived.append(target.name)
        changed_formal_world |= name == "train"
    if changed_formal_world:
        for name in ["judge_log", "rl_bundle", "evaluation", "interim_step5", "comparison.json", "REPORT.md"]:
            path = ROOT / name
            if path.exists():
                target = path.with_name(path.name + ".previous_world." + stamp)
                path.rename(target)
                archived.append(target.name)
    record = dict(prepared_utc=stamp, gpus=4, global_batch=16, accumulation_per_gpu=4,
                  steps=10, archived=archived, launcher_pid=os.getppid(),
                  note="World-size changes restart from the same RT-1 SFT. Calibration and content-addressed API receipts are retained.")
    (ROOT / "4gpu_preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    lock.close()


if __name__ == "__main__":
    main()
