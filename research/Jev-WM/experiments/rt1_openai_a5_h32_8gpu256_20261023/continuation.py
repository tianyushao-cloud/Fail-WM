"""Audited step14 restart with 256 API workers; keep strict resume checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "rt1_openai_a5_h32_8gpu100_20261023"
SOURCE_STEP = 14
WORLD = 8
TARGET = 100
SOURCE_FINGERPRINT = "b67d3fad9316750e1831404bd264144a0bb4ed74b76c95f1f2382afb7e2a91e8"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(config):
    from futureworlds.protocol import implementation, text_identity
    value = {k: v for k, v in config.items() if k not in ("output", "resume")}
    value.update(implementation=implementation(),
                 text_assets=text_identity(config["bundle"], config["dataset"], config.get("text_model")),
                 train_manifest_sha256=digest(config["train_manifest"]),
                 bundle_manifest_sha256=digest(Path(config["bundle"]) / "manifest.json"),
                 world_size=WORLD, schedule_sha256=digest(config["schedule"]))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), value


def validate_change(old, new):
    allowed = {"output", "schedule", "openai_judge", "resume"}
    for key in set(old) | set(new):
        if key not in allowed and old.get(key) != new.get(key):
            raise ValueError("Unapproved continuation change: " + key)
    if (new["steps"], new["global_batch"], new["cpu_threads"],
            new["independent_reference"], new["checkpoint_every"]) != (100, 16, 1, True, 1):
        raise ValueError("Continuation requires total100/global16/threads1/independent reference/checkpoint1")
    if old["method"] != "memspo" or old["initial_variant"] != "sft" or old["reward"] != "openai_gt":
        raise ValueError("Wrong source training protocol")
    if digest(old["schedule"]) != digest(new["schedule"]):
        raise ValueError("Training schedule changed")
    if len(read(new["schedule"])) < TARGET * new["global_batch"]:
        raise ValueError("Training schedule is too short")
    operations = {"workers", "spacing_seconds", "review_dir", "log_dir", "calibration_file"}
    for key in set(old["openai_judge"]) | set(new["openai_judge"]):
        if key not in operations and old["openai_judge"].get(key) != new["openai_judge"].get(key):
            raise ValueError("Judge scoring protocol changed: " + key)
    judge = new["openai_judge"]
    if judge["workers"] != 32 or judge["spacing_seconds"] != 0.02 or judge["max_http_attempts"] != 120000:
        raise ValueError("Expected 256 aggregate API workers, max50 starts/s, retained 120000 attempt budget")
    if digest(judge["calibration_file"]) != judge["calibration_sha256"]:
        raise ValueError("Calibration receipt changed")
    calibration = read(judge["calibration_file"])
    if not calibration["launchable"] or calibration["weights"] != judge["weights"]:
        raise ValueError("Calibration is not launchable")


def verified_checkpoint(folder, expected_fingerprint=None):
    folder = Path(folder)
    if not re.fullmatch(r"step_\d{6}", folder.name):
        raise ValueError("Not a committed checkpoint: " + folder.name)
    receipt = read(folder / "experiment.json")
    if int(folder.name[5:]) != receipt["step"]:
        raise ValueError("Checkpoint directory/step mismatch")
    if expected_fingerprint and receipt["fingerprint"] != expected_fingerprint:
        raise ValueError("Checkpoint protocol mismatch")
    required = {"model.safetensors", "text_adapters.safetensors", "config.json", "training_state.pt"}
    if not required <= set(receipt["files"]):
        raise ValueError("Incomplete checkpoint receipt")
    for name, expected in receipt["files"].items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError("Checkpoint checksum mismatch: " + name)
    return receipt


def latest_checkpoint(output, expected_fingerprint):
    folders = [p for p in (Path(output) / "checkpoints").glob("step_*")
               if p.is_dir() and re.fullmatch(r"step_\d{6}", p.name)]
    if not folders:
        raise RuntimeError("No committed continuation checkpoint; run prepare first")
    folder = max(folders, key=lambda p: int(p.name[5:]))
    receipt = verified_checkpoint(folder, expected_fingerprint)
    return folder, receipt["step"]


def state_digest(value):
    """Storage-independent semantic hash, including exact optimizer tensor bytes."""
    import numpy as np
    import torch
    h = hashlib.sha256()

    def visit(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous()
            h.update(b"tensor" + str(a.dtype).encode() + str(tuple(a.shape)).encode())
            h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(b"array" + str(v.dtype).encode() + str(v.shape).encode() + v.tobytes())
        elif isinstance(v, dict):
            h.update(b"dict")
            for k in sorted(v, key=lambda x: (str(type(x)), str(x))):
                visit(k)
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode() + str(len(v)).encode())
            for x in v:
                visit(x)
        else:
            h.update(repr((type(v).__name__, v)).encode())
    visit(value)
    return h.hexdigest()


def verify_setup(cfg):
    source = OLD / "train/checkpoints/step_000014"
    receipt = verified_checkpoint(source, SOURCE_FINGERPRINT)
    old_run = read(OLD / "train/run.json")
    if old_run["fingerprint"] != SOURCE_FINGERPRINT or hashlib.sha256(
            json.dumps(old_run["config"], sort_keys=True).encode()).hexdigest() != SOURCE_FINGERPRINT:
        raise ValueError("Original training run fingerprint is invalid")
    if old_run["config"]["world_size"] != 8 or receipt["step"] != 14:
        raise ValueError("Expected committed eight-GPU step14")
    validate_change(receipt["config"], cfg)
    # This restart changes only API concurrency and artifact paths, not code.
    manifest = read(ROOT / "source_hashes.json")
    package = ROOT / "source/src/futureworlds"
    actual = {p.relative_to(package).as_posix(): digest(p) for p in package.rglob("*.py")}
    if actual != manifest:
        raise ValueError("Prepared source changed; create a newly audited migration")
    changed = [name for name in actual if actual[name] != digest(OLD / "source/src/futureworlds" / name)]
    if changed:
        raise ValueError("Unexpected source differences: " + repr(changed))
    for name, expected in read(ROOT / "calibration/protocol.json")["source"].items():
        if digest(package / name) != expected:
            raise ValueError("Judge source changed since calibration: " + name)
    fp, effective = fingerprint(cfg)
    if effective["implementation"]["versions"] != old_run["config"]["implementation"]["versions"]:
        raise ValueError("Runtime packages changed from original training environment")
    for key in ("text_assets", "train_manifest_sha256", "bundle_manifest_sha256", "schedule_sha256"):
        if effective[key] != old_run["config"][key]:
            raise ValueError("Source assets changed: " + key)
    from futureworlds.checkpoints import verify_bundle
    from futureworlds.data import disjoint, read_manifest
    verify_bundle(cfg["bundle"])
    train = read_manifest(cfg["train_manifest"], "rt1")
    val = read_manifest(ROOT.parent / "rt1_memspo200_upstream_20261017/data/validation/manifest.json", "rt1")
    test = read_manifest(cfg["evaluation_manifest"], "rt1")
    for a, b in ((train, val), (train, test), (val, test)):
        disjoint(a, b)
    return source, receipt, fp, effective


def prepare(cfg):
    import torch
    source, receipt, fp, effective = verify_setup(cfg)
    output = Path(cfg["output"])
    target = output / "checkpoints/step_000014"
    if target.exists():
        verified_checkpoint(target, fp)
        audit = read(ROOT / "migration.json")
        if audit["source_files"] != receipt["files"] or audit["target_fingerprint"] != fp:
            raise ValueError("Existing migration audit mismatch")
        latest_checkpoint(output, fp)
        print("Existing audited continuation is ready; no state reset", flush=True)
        return fp
    if (output / "run.json").exists() or (output / "complete.json").exists():
        raise RuntimeError("Refusing to overwrite an existing training output")
    state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False)
    if (state["step"], state["world_size"], state["fingerprint"]) != (14, 8, SOURCE_FINGERPRINT):
        raise ValueError("Original optimizer state identity mismatch")
    optimizer_hash = state_digest(state["optimizer"])
    rng_hash = state_digest(state["rng"])
    if len(state["rng"]) != 8:
        raise ValueError("Step14 checkpoint must contain eight rank RNG streams")
    state.update(fingerprint=fp)
    if optimizer_hash != state_digest(state["optimizer"]) or rng_hash != state_digest(state["rng"]):
        raise AssertionError("State conversion changed optimizer or existing RNG streams")
    tmp = target.with_name(target.name + ".partial")
    if tmp.exists():
        tmp.rename(tmp.with_name(tmp.name + ".abandoned." + str(time.time_ns())))
    tmp.mkdir(parents=True)
    for name in receipt["files"]:
        if name != "training_state.pt":
            # Copy immutable weights; target and original must not share writable inodes.
            shutil.copy2(source / name, tmp / name)
    torch.save(state, tmp / "training_state.pt")
    restored = torch.load(tmp / "training_state.pt", map_location="cpu", weights_only=False)
    if state_digest(restored["optimizer"]) != optimizer_hash or state_digest(restored["rng"]) != rng_hash:
        raise AssertionError("Serialization changed optimizer or RNG state")
    for name in ("model.safetensors", "text_adapters.safetensors"):
        if digest(tmp / name) != receipt["files"][name]:
            raise AssertionError("Migrated actor weights differ")
    audit = dict(source_checkpoint=str(source), source_step=14, target_step=100,
                 additional_steps=86, source_world_size=8, target_world_size=8,
                 source_fingerprint=SOURCE_FINGERPRINT, target_fingerprint=fp,
                 source_files=receipt["files"], optimizer_semantic_sha256=optimizer_hash,
                 optimizer_preserved=True, actor_weights_identical=True,
                 all_eight_rng_preserved=True, rng_semantic_sha256=rng_hash,
                 next_step=15, next_schedule_cursor=224,
                 source_history=read(OLD / "migration.json"),
                 source_code_identical=True, global_api_workers_before=16, global_api_workers_after=256,
                 request_starts_per_second_before=4, request_starts_per_second_after=50,
                 reference="same independently loaded original RT-1 SFT reference",
                 equivalence="Actor, Adam state, all eight rank RNG streams and scoring code preserved; only API concurrency and artifact paths change",
                 config_changes={k: dict(before=receipt["config"].get(k), after=cfg.get(k))
                                 for k in sorted(set(receipt["config"]) | set(cfg))
                                 if receipt["config"].get(k) != cfg.get(k)})
    write(tmp / "migration.json", audit)
    write(tmp / "experiment.json", dict(step=14, config=cfg, fingerprint=fp,
          core_sha256=digest(tmp / "model.safetensors"),
          files={p.name: digest(p) for p in tmp.iterdir() if p.is_file()}))
    metrics = [json.loads(x) for x in (OLD / "train/metrics.jsonl").read_text().splitlines() if x.strip()]
    committed = {r["step"]: r for r in metrics if r["step"] <= 14}
    if sorted(committed) != list(range(1, 15)) or not all(r["optimizer_updated"] for r in committed.values()):
        raise ValueError("Original history does not contain fourteen committed optimizer updates")
    (output / "metrics.jsonl").write_text("".join(json.dumps(committed[s]) + "\n" for s in range(1, 15)))
    write(output / "run.json", dict(fingerprint=fp, config=effective))
    write(ROOT / "migration.json", audit)
    tmp.replace(target)
    verified_checkpoint(target, fp)
    print("Migrated step14: actor/Adam/all8 RNG verified; next15, target100, total API workers256", flush=True)
    return fp


def clean_uncommitted(output, committed_step):
    output = Path(output)
    archived = output / "recovery_archive"
    for path in (output / "checkpoints").glob("step_*.partial"):
        archived.mkdir(exist_ok=True)
        path.rename(archived / (path.name + "." + str(time.time_ns())))
    path = output / "metrics.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    keep = {r["step"]: r for r in rows if r["step"] <= committed_step}
    if sorted(keep) != list(range(1, committed_step + 1)):
        raise ValueError("Committed metric history is incomplete")
    if len(rows) != len(keep):
        archived.mkdir(exist_ok=True)
        shutil.copy2(path, archived / ("metrics." + str(time.time_ns()) + ".jsonl"))
        tmp = path.with_name("metrics.jsonl.tmp")
        tmp.write_text("".join(json.dumps(keep[s]) + "\n" for s in sorted(keep)))
        tmp.replace(path)
