"""Audited 4 -> 8 rank continuation; never relax the trainer's resume checks."""
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
OLD = ROOT.parent / "rt1_openai_a5_h32_20261021"
SOURCE_STEP = 10
WORLD = 8
TARGET = 100
SOURCE_FINGERPRINT = "31dcb296778c95571414421294d6cbfff1d6469601e664583d30570ce6866a89"


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
    allowed = {"steps", "output", "schedule", "openai_judge", "cpu_threads",
               "independent_reference", "interim_step", "resume"}
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
    operations = {"workers", "max_http_attempts", "review_dir", "log_dir", "calibration_file"}
    for key in set(old["openai_judge"]) | set(new["openai_judge"]):
        if key not in operations and old["openai_judge"].get(key) != new["openai_judge"].get(key):
            raise ValueError("Judge scoring protocol changed: " + key)
    judge = new["openai_judge"]
    if judge["workers"] != 2 or judge["max_http_attempts"] != 120000:
        raise ValueError("Expected 16 aggregate API workers and 120000 new-run attempt budget")
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


def migrated_rng(source_rng, source_hash):
    import numpy as np
    import torch
    if len(source_rng) != 4:
        raise ValueError("Expected four source rank RNG states")
    states, seeds = copy.deepcopy(source_rng), {}
    for rank in range(4, WORLD):
        seed = int(hashlib.sha256(f"{source_hash}:rank:{rank}".encode()).hexdigest()[:15], 16)
        cuda = torch.Generator(device="cuda:0").manual_seed(seed).get_state().cpu()
        states.append(dict(torch=torch.Generator().manual_seed(seed).get_state(),
                           numpy=np.random.RandomState(seed % 2**32).get_state(),
                           python=random.Random(seed).getstate(),
                           generator=cuda.clone(), cuda=cuda.clone()))
        seeds[str(rank)] = seed
    return states, seeds


def verify_setup(cfg):
    source = OLD / "train/checkpoints/step_000010"
    receipt = verified_checkpoint(source, SOURCE_FINGERPRINT)
    old_run = read(OLD / "train/run.json")
    if old_run["fingerprint"] != SOURCE_FINGERPRINT or hashlib.sha256(
            json.dumps(old_run["config"], sort_keys=True).encode()).hexdigest() != SOURCE_FINGERPRINT:
        raise ValueError("Original training run fingerprint is invalid")
    if old_run["config"]["world_size"] != 4 or receipt["step"] != 10:
        raise ValueError("Expected completed four-GPU step10")
    validate_change(receipt["config"], cfg)
    # Pin the isolated source; only training.py differs from the completed run.
    manifest = read(ROOT / "source_hashes.json")
    package = ROOT / "source/src/futureworlds"
    actual = {p.relative_to(package).as_posix(): digest(p) for p in package.rglob("*.py")}
    if actual != manifest:
        raise ValueError("Prepared source changed; create a newly audited migration")
    changed = [name for name in actual if actual[name] != digest(OLD / "source/src/futureworlds" / name)]
    if changed != ["training.py"]:
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
    target = output / "checkpoints/step_000010"
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
    if (state["step"], state["world_size"], state["fingerprint"]) != (10, 4, SOURCE_FINGERPRINT):
        raise ValueError("Original optimizer state identity mismatch")
    optimizer_hash = state_digest(state["optimizer"])
    old_rng_hash = [state_digest(x) for x in state["rng"]]
    rng, seeds = migrated_rng(state["rng"], receipt["files"]["training_state.pt"])
    state.update(rng=rng, world_size=WORLD, fingerprint=fp)
    if optimizer_hash != state_digest(state["optimizer"]) or old_rng_hash != [state_digest(x) for x in rng[:4]]:
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
    if state_digest(restored["optimizer"]) != optimizer_hash:
        raise AssertionError("Optimizer serialization changed state")
    for name in ("model.safetensors", "text_adapters.safetensors"):
        if digest(tmp / name) != receipt["files"][name]:
            raise AssertionError("Migrated actor weights differ")
    audit = dict(source_checkpoint=str(source), source_step=10, target_step=100,
                 additional_steps=90, source_world_size=4, target_world_size=8,
                 source_fingerprint=SOURCE_FINGERPRINT, target_fingerprint=fp,
                 source_files=receipt["files"], optimizer_semantic_sha256=optimizer_hash,
                 optimizer_preserved=True, actor_weights_identical=True,
                 first_four_rng_preserved=True, new_rank_seeds=seeds,
                 next_step=11, next_schedule_cursor=160,
                 reference="independently loaded original RT-1 SFT; deepcopy hook binding corrected",
                 equivalence="Continuation preserves step10 actor and Adam state; rank resharding and reference correction are not bitwise equivalent to the old four-GPU implementation",
                 config_changes={k: dict(before=receipt["config"].get(k), after=cfg.get(k))
                                 for k in sorted(set(receipt["config"]) | set(cfg))
                                 if receipt["config"].get(k) != cfg.get(k)})
    write(tmp / "migration.json", audit)
    write(tmp / "experiment.json", dict(step=10, config=cfg, fingerprint=fp,
          core_sha256=digest(tmp / "model.safetensors"),
          files={p.name: digest(p) for p in tmp.iterdir() if p.is_file()}))
    metrics = [json.loads(x) for x in (OLD / "train/metrics.jsonl").read_text().splitlines() if x.strip()]
    committed = {r["step"]: r for r in metrics if r["step"] <= 10}
    if sorted(committed) != list(range(1, 11)) or not all(r["optimizer_updated"] for r in committed.values()):
        raise ValueError("Original history does not contain ten committed optimizer updates")
    (output / "metrics.jsonl").write_text("".join(json.dumps(committed[s]) + "\n" for s in range(1, 11)))
    write(output / "run.json", dict(fingerprint=fp, config=effective))
    write(ROOT / "migration.json", audit)
    tmp.replace(target)
    verified_checkpoint(target, fp)
    print("Migrated step10: actor and Adam state verified, next update11, target100, world8", flush=True)
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
