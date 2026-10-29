"""RT-1 SFT -> MemSPO200 with a pure local Qwen-VL reward (no R0 anywhere).

Stages: check -> distributed 1-step smoke -> 200 scheduled batches -> export ->
evaluate SFT and RL on the same validation/test windows at 10/20/32 frames -> report.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
BASE = ROOT.parent / "rt1_memspo200_upstream_20261017"
CALIBRATION = ROOT.parent / "rt1_qwen_judge_calibration_20261019"


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    p = Path(p)
    tmp = p.with_suffix(".partial")
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(p)


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


ALLOWED_CHANGES = {"src/futureworlds/training.py"}
ADDED = {"src/futureworlds/qwen_judge.py", "src/futureworlds/qwen_gt_reward.py"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["check", "all"], default="all", nargs="?")
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()
    cfg = read(ROOT / "train.json")
    baseline = read(BASE / "train.json")
    for k in ["dataset", "method", "initial_variant", "steps", "global_batch", "lr", "adapter_lr", "warmup_steps",
              "horizon", "clip", "kl_coef", "diversity_penalty", "checkpoint_every", "seed", "token_cache",
              "train_manifest", "evaluation_manifest", "bundle", "text_model"]:
        assert cfg[k] == baseline[k], k
    assert cfg["reward"] == "qwen_gt" and "decisions" not in cfg
    assert digest(cfg["schedule"]) == digest(baseline["schedule"])
    if cfg["global_batch"] % args.gpus:
        raise ValueError("global_batch must be divisible by --gpus")
    # Inference, search and metric code must be byte-identical to the upstream
    # baseline; only training.py's reward hook differs, plus the judge modules.
    upstream = read(ROOT / "upstream_source_hashes.json")
    changed = sorted(n for n, h in upstream.items() if digest(SOURCE / n) != h)
    assert set(changed) <= ALLOWED_CHANGES, changed
    assert all((SOURCE / n).is_file() for n in ADDED)
    # Judge settings must be exactly the calibrated, confirmed ones.
    confirm = read(CALIBRATION / "results" / cfg["qwen_judge"]["calibration"] / "summary.json")
    if not confirm["all_gates_passed"]:
        raise RuntimeError("Qwen judge calibration did not pass preset gates; RL not started")
    qj = cfg["qwen_judge"]
    assert Path(confirm["model"]) == Path(qj["model"]) and confirm["scale"] == qj["scale"]
    assert digest(SOURCE / "src/futureworlds/qwen_judge.py") == digest(CALIBRATION / "qwen_judge.py")
    assert abs(qj["tie_tau"] - confirm["gates"]["informative_groups"]["tau"]) < 1e-12
    sys.path.insert(0, str(SOURCE / "src"))
    from futureworlds.checkpoints import verify_bundle
    from futureworlds.data import disjoint, read_manifest

    train = read_manifest(cfg["train_manifest"], "rt1")
    val = read_manifest(BASE / "data/validation/manifest.json", "rt1")
    test = read_manifest(cfg["evaluation_manifest"], "rt1")
    disjoint(train, val)
    disjoint(train, test)
    disjoint(val, test)
    schedule = read(cfg["schedule"])
    assert len(schedule) == 3200 and {x["id"] for x in schedule} <= {x["id"] for x in train["cases"]}
    verify_bundle(cfg["bundle"])
    info = dict(gpus=args.gpus, accumulation_per_gpu=cfg["global_batch"] // args.gpus, scheduled_batches=cfg["steps"],
                global_batch=cfg["global_batch"], horizon=cfg["horizon"], initial_variant="sft",
                reward="pure GT-referenced local Qwen-VL judge; no R0 reward or R0 selection",
                judge=qj, calibration=dict(name=qj["calibration"], gates=confirm["gates"]),
                changed_source_files=changed, source_verified=True,
                caveat="groups with spread <= tie_tau get zero advantage; all-tied global batches skip the optimizer")
    print(json.dumps(info, indent=2), flush=True)
    write(ROOT / "preparation_check.json", info)
    if args.stage == "check":
        return
    import torch

    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f"Need {args.gpus} visible GPUs, found {torch.cuda.device_count()}; no fallback")
    write(ROOT / "launch.json", dict(**info, pid=os.getpid(),
                                     started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    env = {**os.environ, "PYTHONPATH": str(SOURCE / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "USE_TF": "0", "USE_FLAX": "0", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}

    def command(stage, params, distributed=True):
        prefix = [sys.executable, "-u"]
        if distributed:
            prefix += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}"]
        cmd = prefix + ["-m", "futureworlds"] + [str(x) for x in params]
        write(ROOT / "status.json", dict(stage=stage, command=cmd, pid=os.getpid()))
        print("STAGE", stage, flush=True)
        with (ROOT / "logs" / f"{stage}.log").open("a") as stream:
            subprocess.run(cmd, cwd=SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def training(stage, config):
        c = read(config)
        out = Path(c["output"])
        if (out / "complete.json").exists():
            return
        checkpoints = sorted((out / "checkpoints").glob("step_*/experiment.json"))
        if checkpoints:
            c["resume"] = str(checkpoints[-1].parent)
            config = ROOT / f"{stage}_resume.json"
            write(config, c)
        elif (out / "run.json").exists():
            # No optimizer state persisted yet: restart from the same SFT, keep evidence.
            out.rename(out.with_name(out.name + ".incomplete." + str(time.time_ns())))
        command(stage, ["train", "--config", config])

    training("distributed_smoke", ROOT / "smoke.json")
    smoke = [json.loads(x) for x in (ROOT / "smoke/train/metrics.jsonl").read_text().splitlines()]
    if not any(x["optimizer_updated"] for x in smoke):
        raise RuntimeError("Distributed smoke produced no informative groups; formal training not started")
    training("train", ROOT / "train.json")
    command("export", ["export-model", "--bundle", cfg["bundle"], "--dataset", "rt1", "--checkpoint",
                       ROOT / f"train/checkpoints/step_{cfg['steps']:06d}", "--output", ROOT / "rl_bundle"],
            distributed=False)
    for split in ["validation", "test"]:
        for label, bundle, variant in [("sft", cfg["bundle"], "sft"), ("rl", ROOT / "rl_bundle", "main")]:
            command(f"evaluate_{split}_{label}", ["evaluate", "--bundle", bundle, "--dataset", "rt1", "--variant", variant,
                    "--manifest", BASE / f"data/{split}/manifest.json", "--output", ROOT / f"evaluation/{split}/{label}",
                    "--device", "cuda", "--text-model", cfg["text_model"], "--horizons", 10, 20, 32, "--beam-width", 4],
                    distributed=False)
    subprocess.run([sys.executable, str(ROOT / "report.py")], cwd=ROOT, env=env, check=True)
    write(ROOT / "status.json", dict(stage="COMPLETE", utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == "__main__":
    lock = (ROOT / ".launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except Exception as error:
        write(ROOT / "status.json", dict(stage="FAILED", error=repr(error)))
        raise
