"""RT-1 SFT -> Decisions multi-axis MemSPO; retain Qwen rollout and optimizer settings."""
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
QWEN = ROOT.parent / "rt1_qwen_a5_h32_20261020"
BASE = ROOT.parent / "rt1_memspo200_upstream_20261017"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["check", "probe", "calibrate", "pilot", "all"], default="check", nargs="?")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    cfg, baseline = read(ROOT / "train.json"), read(QWEN / "train.json")
    for key in ["dataset", "method", "bundle", "initial_variant", "seed", "global_batch", "lr",
                "adapter_lr", "warmup_steps", "horizon", "clip", "kl_coef", "diversity_penalty",
                "train_manifest", "evaluation_manifest", "text_model"]:
        assert cfg[key] == baseline[key], key
    assert cfg["reward"] == "openai_gt" and cfg["initial_variant"] == "sft"
    assert 1 <= cfg["steps"] <= baseline["steps"]
    assert cfg["steps"] % cfg["checkpoint_every"] == 0
    assert not cfg.get("initial_checkpoint"), "This run must start from the RT-1 SFT bundle"
    requested = cfg["openai_judge"].get("requested_weights", cfg["openai_judge"]["weights"])
    assert requested == baseline["qwen_judge"]["weights"]
    if cfg["openai_judge"]["weights"] != requested:
        selection = read(cfg["openai_judge"]["calibration_file"])
        assert selection["launchable"] and selection["requested_weights"] == requested
        assert selection["weights"] == cfg["openai_judge"]["weights"]
    assert digest(cfg["schedule"]) == digest(baseline["schedule"])
    assert cfg["global_batch"] % args.gpus == 0
    changed = []
    for path in sorted((QWEN / "source/src/futureworlds").rglob("*.py")):
        rel = path.relative_to(QWEN / "source")
        if digest(path) != digest(SOURCE / rel):
            changed.append(str(rel))
    assert changed == ["src/futureworlds/training.py"], changed
    sys.path.insert(0, str(SOURCE / "src"))
    from futureworlds.checkpoints import verify_bundle
    from futureworlds.data import disjoint, read_manifest
    from futureworlds.openai_judge import PROTOCOL
    train = read_manifest(cfg["train_manifest"], "rt1")
    val = read_manifest(BASE / "data/validation/manifest.json", "rt1")
    test = read_manifest(cfg["evaluation_manifest"], "rt1")
    for left, right in [(train, val), (train, test), (val, test)]:
        disjoint(left, right)
    verify_bundle(cfg["bundle"])
    assert {x["id"] for x in read(cfg["schedule"])} <= {x["id"] for x in train["cases"]}
    info = dict(initial_variant="sft", horizon=cfg["horizon"], method=cfg["method"],
                global_batch=cfg["global_batch"], steps=cfg["steps"], gpus=args.gpus,
                accumulation_per_gpu=cfg["global_batch"] // args.gpus,
                protocol=PROTOCOL, changed_existing_source=changed,
                reward="pure Decisions probability scores, original Qwen aggregation over calibrated axes",
                requested_weights=requested, weights=cfg["openai_judge"]["weights"],
                source_verified=True, schedule_sha256=digest(cfg["schedule"]),
                feature_cache="recomputed with the same codec and T5; old source fingerprint is not reused",
                status="prepared; calibration and pilot must pass before formal training")
    write(ROOT / "preparation_check.json", info)
    print(json.dumps(info, indent=2), flush=True)
    if args.stage == "check":
        return
    from runtime import api_environment
    api_environment()
    env = {**os.environ, "PYTHONPATH": str(SOURCE / "src"), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "USE_TF": "0", "USE_FLAX": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}

    def command(stage, params, distributed=False):
        prefix = [sys.executable, "-u"]
        if distributed:
            prefix += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}"]
        cmd = prefix + [str(x) for x in params]
        write(ROOT / "status.json", dict(stage=stage, pid=os.getpid(), command=cmd,
                                         utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print("STAGE", stage, flush=True)
        with (ROOT / "logs" / f"{stage}.log").open("a") as stream:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    if args.stage == "probe":
        command("probe", [ROOT / "calibrate.py", "--probe"])
        return
    calibration = Path(cfg["openai_judge"]["calibration_file"])
    if not calibration.exists():
        command("calibrate", [ROOT / "calibrate.py", "--cases", 4])
    summary = read(calibration)
    if not summary["launchable"]:
        raise RuntimeError("Decisions pilot calibration failed; training not started")
    for name, expected in read(ROOT / "calibration/protocol.json")["source"].items():
        assert digest(SOURCE / "src/futureworlds" / name) == expected, "Judge source changed since calibration"
    cfg["openai_judge"]["calibration_sha256"] = digest(calibration)
    write(ROOT / "train.json", cfg)
    if args.stage == "calibrate":
        return
    import torch
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f"Need {args.gpus} visible GPUs, found {torch.cuda.device_count()}")

    def train(stage, config):
        out = Path(config["output"])
        if (out / "complete.json").exists():
            return
        checkpoints = sorted((out / "checkpoints").glob("step_*/experiment.json"))
        config_path = ROOT / (stage + "_launch.json")
        if checkpoints:
            config["resume"] = str(checkpoints[-1].parent)
        elif (out / "run.json").exists():
            # A failed partial batch has no optimizer state to resume. Preserve
            # it, restart at the same SFT/seed, and replay cached API receipts.
            out.rename(out.with_name(out.name + ".incomplete." + str(time.time_ns())))
        write(config_path, config)
        command(stage, ["-m", "futureworlds", "train", "--config", config_path], distributed=args.gpus > 1)

    pilot = json.loads(json.dumps(cfg))
    pilot.update(steps=1, checkpoint_every=1, output=str(ROOT / "pilot/train"))
    pilot["openai_judge"].update(max_http_attempts=4096, log_dir=str(ROOT / "pilot/judge_log"))
    train("pilot", pilot)
    metrics = [json.loads(x) for x in (ROOT / "pilot/train/metrics.jsonl").read_text().splitlines()]
    if not any(x["optimizer_updated"] for x in metrics):
        raise RuntimeError("Pilot had no informative groups; formal training not started")
    if args.stage == "pilot":
        write(ROOT / "status.json", dict(stage="PILOT_COMPLETE", optimizer_updates=1))
        return
    train("train", cfg)
    # A fixed halfway checkpoint shows validation trend; never select by test.
    middle = cfg["checkpoint_every"]
    if middle < cfg["steps"]:
        command("export_interim", ["-m", "futureworlds", "export-model", "--bundle", cfg["bundle"],
                "--dataset", "rt1", "--checkpoint", ROOT / f"train/checkpoints/step_{middle:06d}",
                "--output", ROOT / f"interim_step{middle}/rl_bundle"])
        command("evaluate_validation_interim", ["-m", "futureworlds", "evaluate", "--bundle",
                ROOT / f"interim_step{middle}/rl_bundle", "--dataset", "rt1", "--variant", "main",
                "--manifest", BASE / "data/validation/manifest.json", "--output",
                ROOT / f"evaluation/validation/step{middle}", "--device", "cuda",
                "--text-model", cfg["text_model"], "--horizons", 10, 20, 32, "--beam-width", 4])
    command("export", ["-m", "futureworlds", "export-model", "--bundle", cfg["bundle"], "--dataset", "rt1",
                       "--checkpoint", ROOT / f"train/checkpoints/step_{cfg['steps']:06d}",
                       "--output", ROOT / "rl_bundle"])
    for split in ["validation", "test"]:
        for label, bundle, variant in [("sft", cfg["bundle"], "sft"), ("rl", ROOT / "rl_bundle", "main")]:
            command(f"evaluate_{split}_{label}", ["-m", "futureworlds", "evaluate", "--bundle", bundle,
                    "--dataset", "rt1", "--variant", variant, "--manifest", BASE / f"data/{split}/manifest.json",
                    "--output", ROOT / f"evaluation/{split}/{label}", "--device", "cuda",
                    "--text-model", cfg["text_model"], "--horizons", 10, 20, 32, "--beam-width", 4])
    command("report", [ROOT / "report.py"])
    write(ROOT / "status.json", dict(stage="COMPLETE", steps=cfg["steps"]))


if __name__ == "__main__":
    lock = (ROOT / ".launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except Exception as error:
        write(ROOT / "status.json", dict(stage="FAILED", error=repr(error)))
        raise
