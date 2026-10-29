"""Eight-GPU continuation of the completed four-GPU RT-1 Decisions run."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from continuation import (ROOT, OLD, TARGET, WORLD, read, write, prepare,
                          verify_setup, latest_checkpoint, clean_uncommitted)

SOURCE = ROOT / "source"
BASE = ROOT.parent / "rt1_memspo200_upstream_20261017"
WAIT_SECONDS = 48 * 3600
sys.path.insert(0, str(SOURCE / "src"))


def status(stage, **details):
    write(ROOT / "status.json", dict(stage=stage, pid=os.getpid(),
          utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details))


def environment():
    return {**os.environ, "PYTHONPATH": str(SOURCE / "src"), "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "USE_TF": "0", "USE_FLAX": "0",
            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}


def command(stage, params, distributed=False):
    prefix = [sys.executable, "-u"]
    if distributed:
        prefix += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={WORLD}"]
    cmd = prefix + [str(x) for x in params]
    status(stage, command=cmd)
    print("STAGE", stage, flush=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "logs" / f"{stage}.log").open("a") as stream:
        subprocess.run(cmd, cwd=ROOT, env=environment(), stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def network_ready(deadline=None):
    from runtime import wait_for_api
    remaining = WAIT_SECONDS if deadline is None else deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("48h recovery window expired; committed checkpoints were preserved")
    wait_for_api(max_wait_seconds=remaining,
                 on_wait=lambda record: status(**record))


def network_failure(text):
    # Never retry invalid scores/credentials, exhausted budgets, or model errors.
    fatal = ("HTTP 400", "HTTP 401", "HTTP 403", "budget exhausted", "OutOfMemoryError",
             "out of memory", "checksum mismatch", "Nonfinite", "Cached receipt identity mismatch",
             "probabilities", "scoring response", "Resume protocol/world-size mismatch")
    transient = ("Official Decisions transport failed", "RemoteDisconnected",
                 "Official Decisions HTTP 408", "Official Decisions HTTP 429",
                 "Official Decisions HTTP 500", "Official Decisions HTTP 502",
                 "Official Decisions HTTP 503", "Official Decisions HTTP 504")
    return not any(x in text for x in fatal) and any(x in text for x in transient)


def train(config, fp):
    output = Path(config["output"])
    recovery_deadline = time.monotonic() + WAIT_SECONDS
    previous = 10
    failures = 0
    while True:
        checkpoint, committed = latest_checkpoint(output, fp)
        if committed < 10 or committed > TARGET:
            raise ValueError("Unexpected continuation step")
        if committed > previous:
            recovery_deadline = time.monotonic() + WAIT_SECONDS
            previous = committed
            failures = 0
        clean_uncommitted(output, committed)
        marker = output / "complete.json"
        if marker.exists():
            done = read(marker)
            if done.get("fingerprint") != fp or done.get("steps") != TARGET or committed != TARGET:
                raise ValueError("Invalid completion marker; refusing to skip training")
            return
        launch = json.loads(json.dumps(config))
        launch["resume"] = str(checkpoint)
        config_path = ROOT / "train_launch.json"
        write(config_path, launch)
        log = ROOT / "logs/train.log"
        offset = log.stat().st_size if log.exists() else 0
        try:
            command("train", ["-m", "futureworlds", "train", "--config", config_path], distributed=True)
            # A successful child must have published a valid step100/complete marker.
            if not marker.exists():
                raise RuntimeError("Training child exited without a completion marker")
        except subprocess.CalledProcessError:
            with log.open("rb") as stream:
                stream.seek(max(offset, log.stat().st_size - 1024 * 1024))
                failure = stream.read().decode(errors="replace")
            if not network_failure(failure):
                raise
            _, recovered = latest_checkpoint(output, fp)
            if recovered > previous:
                recovery_deadline = time.monotonic() + WAIT_SECONDS
                previous, failures = recovered, 0
            failures += 1
            remaining = recovery_deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Decisions scoring did not recover within 48h; checkpoints preserved")
            delay = min(30 * 2 ** min(failures - 1, 4), 300, remaining)
            status("NETWORK_RETRY", committed_step=recovered, next_step=recovered + 1,
                   retry=failures, wait_seconds=delay, recovery_limit_seconds=WAIT_SECONDS)
            print(f"NETWORK_RETRY: step{recovered} committed; wait {delay:.0f}s, then resume", flush=True)
            time.sleep(delay)
            network_ready(recovery_deadline)


def finish(config):
    # Preserve the same fixed evaluation protocol. Reuse the completed SFT baselines.
    for middle in [config["interim_step"], config["steps"]]:
        label = "rl" if middle == TARGET else f"step{middle}"
        bundle = ROOT / "rl_bundle" if middle == TARGET else ROOT / f"interim_step{middle}/rl_bundle"
        receipt = ROOT / f"export_step{middle}.json"
        if not receipt.exists():
            command(f"export_step{middle}", ["-m", "futureworlds", "export-model", "--bundle", config["bundle"],
                    "--dataset", "rt1", "--checkpoint", ROOT / f"train/checkpoints/step_{middle:06d}",
                    "--output", bundle])
            write(receipt, dict(step=middle, bundle=str(bundle)))
        splits = ["validation", "test"] if middle == TARGET else ["validation"]
        for split in splits:
            output = ROOT / f"evaluation/{split}/{label}"
            if not (output / "summary.json").exists():
                command(f"evaluate_{split}_{label}", ["-m", "futureworlds", "evaluate", "--bundle", bundle,
                        "--dataset", "rt1", "--variant", "main", "--manifest", BASE / f"data/{split}/manifest.json",
                        "--output", output, "--device", "cuda", "--text-model", config["text_model"],
                        "--horizons", 10, 20, 32, "--beam-width", 4])
    command("report", [ROOT / "report.py"])
    status("COMPLETE", steps=TARGET, inherited_steps=10, new_steps=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["check", "prepare", "train", "all", "finish"], nargs="?", default="check")
    args = parser.parse_args()
    cfg = read(ROOT / "train.json")
    _, _, fp, _ = verify_setup(cfg)
    if args.stage == "check":
        checkpoint, committed = latest_checkpoint(cfg["output"], fp)
        write(ROOT / "preparation_check.json", dict(state="READY", world_size=WORLD,
              target_steps=TARGET, committed_steps=committed, next_step=committed + 1,
              global_batch=16, accumulation_per_rank=2, checkpoint=str(checkpoint),
              wait_seconds=WAIT_SECONDS, fingerprint=fp))
        status("READY", running=False, target_steps=TARGET, committed_steps=committed,
               next_step=committed + 1, world_size=WORLD)
        print(f"READY: 8 GPUs, global batch16, accumulation2, committed{committed}, target100, wait48h", flush=True)
        return
    fp = prepare(cfg)
    if args.stage == "prepare":
        status("PREPARED", running=False, target_steps=TARGET, world_size=WORLD)
        return
    import torch
    if args.stage != "finish" and torch.cuda.device_count() < WORLD:
        raise RuntimeError(f"Need 8 visible GPUs, found {torch.cuda.device_count()}; use the eight-GPU job")
    if args.stage != "finish":
        # If step100 is already committed, no network/API access is required.
        _, step = latest_checkpoint(cfg["output"], fp)
        if step < TARGET:
            from runtime import api_environment
            api_environment()
            network_ready()
        train(cfg, fp)
    if args.stage in ("all", "finish"):
        finish(cfg)
    else:
        status("TRAIN_COMPLETE", steps=TARGET)


if __name__ == "__main__":
    lock = (ROOT / ".launcher.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This eight-GPU continuation already has an active launcher")
    try:
        main()
    except Exception as error:
        status("FAILED", error=repr(error))
        raise
