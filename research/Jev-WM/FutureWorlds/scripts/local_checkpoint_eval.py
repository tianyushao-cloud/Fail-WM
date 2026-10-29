"""Validation-only checkpoint selection and a distributed horizon32 smoke test."""
import json, math, os
from pathlib import Path
from success_memspo import launch, stable_json
from futureworlds.checkpoints import file_sha256
from futureworlds.data import read_manifest

STEPS=(50,100,150,200)

def choose_checkpoint(candidates):
    if [x["step"] for x in candidates] != list(STEPS):
        raise ValueError("All four validation checkpoints are required")
    if any(not math.isfinite(x["lpips32"]) for x in candidates):
        raise ValueError("Nonfinite validation score")
    return min(candidates,key=lambda x:(x["lpips32"],x["step"]))

def distributed_smoke(a):
    """One real update with formal world size/batch/horizon; isolated optimizer."""
    out=Path(a.output).resolve()
    cfg=json.loads((out/"train.json").read_text())
    smoke=out/"distributed_smoke"
    cfg.update(steps=1,checkpoint_every=1,output=str(smoke/"train"))
    cfg.pop("resume",None)
    stable_json(smoke/"config.json",cfg)
    complete=smoke/"train/complete.json"
    if complete.exists():
        if json.loads(complete.read_text()).get("steps")!=1:
            raise ValueError("Unexpected distributed smoke completion")
        state=json.loads((smoke/"train/run.json").read_text())["config"]
        if state["world_size"]!=a.gpus or state["horizon"]!=32 or state["global_batch"]!=16:
            raise ValueError("Distributed smoke configuration differs")
        print("Horizon32 distributed smoke already passed",flush=True)
        return
    launch(a,"train",["--config",smoke/"config.json"])

def evaluate_checkpoints(a):
    out=Path(a.output).resolve()
    manifest=out/"data/validation/manifest.json"
    rows=read_manifest(manifest,"robocasa")["cases"]
    ids=[r["id"] for r in rows]
    candidates=[]
    for step in STEPS:
        checkpoint=out/"train/checkpoints"/f"step_{step:06d}"
        bundle=out/"validation_bundles"/f"step_{step:06d}"
        dest=out/"validation_checkpoints"/f"step_{step:06d}"
        if not a.dry_run and not (checkpoint/"experiment.json").exists():
            raise ValueError(f"Missing validation candidate: {checkpoint}")
        launch(a,"export-model",["--bundle",a.bundle,"--dataset","robocasa",
            "--checkpoint",checkpoint,"--output",bundle],False)
        launch(a,"evaluate",["--bundle",bundle,"--dataset","robocasa","--variant","main",
            "--manifest",manifest,"--output",dest,"--device","cuda","--metric-device","cuda",
            "--text-model",a.text_model,"--horizons",10,20,32,"--beam-width",4,"--memory","full"])
        if a.dry_run:
            continue
        summary=json.loads((dest/"summary.json").read_text())
        protocol=summary["protocol"]
        if (summary["state"]!="COMPLETE" or protocol["cases"]!=ids or
            protocol["manifest_sha256"]!=file_sha256(manifest) or
            len(summary["per_case"])!=len(ids)):
            raise ValueError(f"Incomplete/mismatched validation: step {step}")
        if protocol["bundle_manifest_sha256"]!=file_sha256(bundle/"manifest.json"):
            raise ValueError("Validation bundle mismatch")
        candidates.append(dict(step=step,lpips32=summary["aggregate"]["32"]["lpips"],
            summary_sha256=file_sha256(dest/"summary.json"),bundle=str(bundle),evaluation=str(dest)))
    if a.dry_run:
        print("After all four validation evaluations: select lowest mean LPIPS32 "
              "(tie -> earlier step), then evaluate only selected checkpoint on test.",flush=True)
        return
    best=choose_checkpoint(candidates)
    stable_json(out/"checkpoint_selection.json",dict(
        criterion="lowest validation mean LPIPS at 32 generated frames",
        tie_break="earliest step",candidates=candidates,selected_step=best["step"],
        selected_bundle=best["bundle"],test_used_for_selection=False))
    # Stable aliases avoid duplicating selected weights/predictions.
    for alias,target in [(out/"rl_bundle",Path(best["bundle"])),
                         (out/"evaluation/validation/rl",Path(best["evaluation"]))]:
        alias.parent.mkdir(parents=True,exist_ok=True)
        if alias.exists() or alias.is_symlink():
            if alias.resolve()!=target.resolve():
                raise ValueError(f"Selected result alias differs: {alias}")
        else:
            alias.symlink_to(os.path.relpath(target,alias.parent),target_is_directory=True)
    launch(a,"evaluate",["--bundle",out/"rl_bundle","--dataset","robocasa","--variant","main",
        "--manifest",out/"data/test/manifest.json","--output",out/"evaluation/test/rl",
        "--device","cuda","--metric-device","cuda","--text-model",a.text_model,
        "--horizons",10,20,32,"--beam-width",4,"--memory","full"])

