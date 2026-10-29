"""Original MemSPO/R0 from the local success-only RT-1-initialized SFT."""
import argparse,json
from pathlib import Path
from futureworlds.checkpoints import file_sha256,verify_bundle
from futureworlds.data import read_manifest
from scholar_success_memspo import verify_source,prepare
from success_memspo import stable_json,preflight,evaluate,train,report

def audit_sft_splits(a):
    bundle=Path(a.bundle).resolve()
    verify_bundle(bundle)
    p=bundle/"provenance"
    receipt=json.loads((p/"sft_experiment.json").read_text())
    run=json.loads((p/"sft_run.json").read_text())
    done=json.loads((p/"sft_complete.json").read_text())
    if receipt["step"]!=10000 or receipt["config"]["method"]!="sft":
        raise ValueError("Expected local 10000-step SFT")
    if not receipt["config"].get("initial_checkpoint","").endswith("/rt1-upstream/model"):
        raise ValueError("Expected RT-1 initialization")
    if done["state"]!="COMPLETE" or done["fingerprint"]!=receipt["fingerprint"]:
        raise ValueError("SFT completion/receipt mismatch")
    settings=json.loads((bundle/"robocasa/futureworlds_config.json").read_text())
    for name in ["model.safetensors","text_adapters.safetensors"]:
        folder=bundle/settings["variants"]["sft"]["path"]
        if file_sha256(folder/name)!=receipt["files"][name]:
            raise ValueError("Local SFT weight differs from copied receipt")
    codec=json.loads((p/"codec_experiment.json").read_text())
    if codec["step"]!=44000 or file_sha256(bundle/settings["codec"]/"diffusion_pytorch_model.safetensors")!=codec["files"]["diffusion_pytorch_model.safetensors"]:
        raise ValueError("SFT/codec pairing mismatch")
    sft_manifest=Path(receipt["config"]["train_manifest"])
    if file_sha256(sft_manifest)!=run["config"]["train_manifest_sha256"]:
        raise ValueError("Original SFT training manifest changed")
    sft_rows=read_manifest(sft_manifest,"robocasa")["cases"]
    codec_manifest=Path(json.loads((p/"codec_run.json").read_text())["config"]["train_manifest"])
    if file_sha256(codec_manifest)!=file_sha256(sft_manifest):
        raise ValueError("Selected codec was not trained on the same success-only episodes")
    if any(r.get("success") is not True for r in sft_rows):
        raise ValueError("SFT/codec training includes failure data")
    source=Path(a.data_root)
    rows={}
    for split,name in [("train","train_success.json"),("validation","val.json"),("test","test.json")]:
        original=read_manifest(source/name,"robocasa")["cases"]
        rows[split]=[r for r in original if r.get("success") is True]
        if split=="train" and len(rows[split])!=len(original):raise ValueError("Failures in RL train")
    sft_ids={r["episode_id"] for r in sft_rows}
    sft_groups={r["initial_state_id"] for r in sft_rows}
    if {r["episode_id"] for r in rows["train"]}!=sft_ids:
        raise ValueError("RL training cohort differs from selected SFT training cohort")
    checked={}
    for split in ["validation","test"]:
        original=read_manifest(source/("val.json" if split=="validation" else "test.json"),"robocasa")["cases"]
        for key,seen in [("episode_id",sft_ids),("initial_state_id",sft_groups)]:
            if seen & {r[key] for r in original}:raise ValueError(f"SFT/codec training leakage into {split}/{key}")
        checked[split]=dict(success_episodes=len(rows[split]),initial_states=len({r["initial_state_id"] for r in rows[split]}),overlap_with_sft_train_episodes=0,overlap_with_sft_train_initial_states=0,overlap_with_codec_train_episodes=0)
    for key in ["episode_id","initial_state_id"]:
        if {r[key] for r in rows["validation"]}&{r[key] for r in rows["test"]}:raise ValueError("Validation/test overlap")
    result=dict(initial_model="RT-1 -> local success-only SFT 10000",sft_checkpoint_sha256=file_sha256(p/"sft_experiment.json"),codec_checkpoint_step=44000,codec_train_success=len(sft_rows),codec_train_failure=0,sft_train_success=len(sft_rows),sft_train_failure=0,rl_train_success=len(rows["train"]),rl_train_failure=0,rl_and_sft_training_episode_overlap=len(sft_ids),heldout_checks=checked,bundle=str(bundle),bundle_manifest_sha256=file_sha256(bundle/"manifest.json"),limitations="Validation used for codec checkpoint selection and prior diagnostics; test also inspected in prior work. Both disjoint from optimizer training, but neither is a new blind evaluation.")
    stable_json(Path(a.output)/"sft_data_audit.json",result)
    print(json.dumps(dict(event="sft_data_audit",**result)),flush=True)
    return result

def assert_episode_start(a):
    """Reject stale interior-start data/checkpoints before launching any stage."""
    out=Path(a.output)
    spec=json.loads((out/"protocol.json").read_text())
    if spec.get("initial_reference_policy")!="original_episode_first_two":
        raise ValueError("Expected original first-two-frame experiment; refusing stale interior-start run")
    if spec["horizon"]!=a.horizon or json.loads((out/"train.json").read_text())["horizon"]!=a.horizon:
        raise ValueError("Requested horizon differs from prepared experiment; choose a distinct output")
    if spec["world_size"]!=a.gpus:
        raise ValueError("Prepared GPU count differs from launch GPU count")
    for split in ["train","validation","test"]:
        manifest=read_manifest(out/"data"/split/"manifest.json","robocasa")
        if any(r.get("source_start")!=0 or r.get("success") is not True or r["frames"]<max(a.horizon,32)+2 for r in manifest["cases"]):
            raise ValueError(f"Interior-start or failure case in {split}")
        if len({r["episode_id"] for r in manifest["cases"]})!=len(manifest["cases"]):
            raise ValueError(f"Expected one original-start window per episode: {split}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",nargs="?",default="all",choices=["prepare","preflight","smoke","baseline","train","evaluate","report","all"])
    for key in ["bundle","data-root","text-model","output"]:p.add_argument("--"+key,required=True)
    p.add_argument("--gpus",type=int,default=8);p.add_argument("--dry-run",action="store_true")
    p.add_argument("--horizon",type=int,choices=[10,32],default=10)
    a=p.parse_args()
    if a.gpus not in (4,8):p.error("Formal experiment supports four or eight GPUs")
    lock=verify_source()
    protocol_path=Path(a.output)/"protocol.json"
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if previous["horizon"]!=a.horizon:
            raise ValueError("Existing run has another horizon; use a distinct OUTPUT, never resume across horizons")
        if previous.get("initial_reference_policy")!="original_episode_first_two":
            raise ValueError("Old interior-start output exists; do not resume it with the new first-two-frame protocol")
        if previous["world_size"]!=a.gpus:
            output=Path(a.output)
            if (output/"train/run.json").exists() or list((output/"train/checkpoints").glob("step_*")):
                raise ValueError("RL has started with another world size; choose a new OUTPUT")
            if a.stage not in ["prepare","all"]:
                raise ValueError("Run prepare before switching GPU count")
            previous["world_size"]=a.gpus
            protocol_path.write_text(json.dumps(previous,indent=2)+"\n")
    recipe=json.loads((Path(__file__).resolve().parents[1]/"vendor/scholar_futureworlds/configs/train/robocasa_memspo.json").read_text())
    a.steps=recipe["steps"];a.seed=recipe["seed"]
    audit=audit_sft_splits(a)
    if a.stage in ["prepare","all"]:prepare(a,lock,episode_start=True,horizon=a.horizon)
    assert_episode_start(a)
    if a.stage in ["preflight","smoke","all"] and not a.dry_run:preflight(a)
    if a.stage in ["smoke","all"] and a.horizon==32:
        from local_checkpoint_eval import distributed_smoke
        distributed_smoke(a)
    if a.stage in ["baseline","all"]:evaluate(a,"sft")
    if a.stage in ["train","all"]:train(a)
    if a.stage in ["evaluate","all"]:
        if a.horizon==32:
            from local_checkpoint_eval import evaluate_checkpoints
            evaluate_checkpoints(a)
        else:evaluate(a,"rl")
    if a.stage in ["report","all"] and not a.dry_run:
        report(a)
        path=Path(a.output)/"REPORT.md"
        text=path.read_text().replace("成功轨迹 MemSPO 与原始 SFT 对照","成功轨迹 MemSPO 与本地适配 SFT 对照").replace("原始 SFT 初始化","RT-1 起点、成功数据 SFT 10000步权重初始化").replace("独立测试集","按初态划分的测试集")
        path.write_text(text+f"\nBase: local success-only SFT step10000; paired success-only codec step44000. SFT and RL reuse the same 443 training episodes. No train episode or initial-state overlap with the 53/59 success validation/test episodes.\nReleased unmodified MemSPO/R0 trainer; 200 updates; configured rollout horizon={a.horizon}; Every trajectory uses its original first two observations (source_start=0); fixed upstream episode sampler. RL consumes first {a.horizon+2} frames ({a.horizon} generated), evaluation first34 (32 generated), no GT refresh.\nValidation used to choose codec, and validation/test have been inspected in earlier experiments; this comparison is not new blind generalization evidence.\n")

if __name__=="__main__":main()

