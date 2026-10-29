"""Released MemSPO implementation and recipe, adapted to local success-only data."""
import argparse, collections, concurrent.futures, hashlib, json
from pathlib import Path
import numpy as np
from futureworlds.data import read_manifest, read_case, write_json
from futureworlds.checkpoints import file_sha256
from futureworlds.sampling import Schedule
from success_memspo import stable_json, preflight, evaluate, train, report
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/"vendor/scholar_futureworlds"

def verify_source():
    lock=json.loads((VENDOR/"SOURCE_LOCK.json").read_text())
    patch_file=VENDOR/"COMPATIBILITY_PATCHES.json"
    patches=json.loads(patch_file.read_text()) if patch_file.exists() else {"files":{}}
    if patches.get("base_commit",lock["commit"])!=lock["commit"]:
        raise ValueError("Compatibility patches target another source revision")
    if set(patches["files"])-set(lock["files"]):
        raise ValueError("Compatibility patch names unknown source files")
    for name,digest in lock["files"].items():
        patch=patches["files"].get(name)
        if patch and patch["original_sha256"]!=digest:
            raise ValueError(f"Compatibility patch base mismatch: {name}")
        expected=patch["patched_sha256"] if patch else digest
        if file_sha256(VENDOR/name)!=expected:raise ValueError(f"Released source changed: {name}")
    import futureworlds
    if Path(futureworlds.__file__).resolve().parent!=VENDOR/"src/futureworlds":
        raise ValueError("Use run_scholar_success_4gpu.sh to select the original source")
    return lock

def materialize(job):
    manifest,row,starts,folder=job
    images,actions,text=read_case(manifest,row,52)
    cases=[]
    for start in sorted(starts):
        cid=f"{row['id']}_s{start:06d}_n042";target=folder/(cid+".npz")
        values=dict(images=images[start:start+42],actions=actions[start:start+42].copy(),instruction=np.array(text))
        values["actions"][-1]=0
        if target.exists():
            with np.load(target,allow_pickle=False) as z:
                if any(not np.array_equal(z[k],v) for k,v in values.items()):
                    raise ValueError(f"Prepared window changed: {cid}")
        else:
            tmp=target.with_suffix(".partial")
            with tmp.open("wb") as f:np.savez_compressed(f,**values)
            tmp.replace(target)
        cases.append(dict(id=cid,episode_id=row["episode_id"],initial_state_id=row["initial_state_id"],
            file=target.name,sha256=file_sha256(target),frames=42,split=row["split"],task=row["task"],
            success=True,source_manifest=str(manifest),source_file=row["file"],
            source_sha256=row["sha256"],source_start=start))
    return cases

def prepare(a,lock,episode_start=False,horizon=None):
    out=Path(a.output).resolve();source=Path(a.data_root).resolve()
    original=json.loads((VENDOR/"configs/train/robocasa_memspo.json").read_text())
    effective_horizon=original["horizon"] if horizon is None else horizon
    if not 1 <= effective_horizon <= 40:
        raise ValueError("42-frame source windows support 1..40 generated frames")
    manifests={};rows={}
    for split,name in [("train","train_success.json"),("validation","val.json"),("test","test.json")]:
        manifests[split]=source/name;allrows=read_manifest(manifests[split],"robocasa")["cases"]
        rows[split]=[r for r in allrows if r.get("success") is True]
        if not rows[split] or any(r["frames"]<42 or not r.get("initial_state_id") for r in rows[split]):
            raise ValueError(f"Missing/short success episode or group in {split}")
        if split=="train" and len(rows[split])!=len(allrows):raise ValueError("Failures in train_success")
    for x,y in [("train","validation"),("train","test"),("validation","test")]:
        for key in ["episode_id","initial_state_id"]:
            if {r[key] for r in rows[x]}&{r[key] for r in rows[y]}:raise ValueError(f"Split leakage: {x}/{y}/{key}")
    virtual=[]
    for row in rows["train"]:
        for start in ([0] if episode_start else range(row["frames"]-42+1)):
            virtual.append(dict(row,id=f"{row['id']}_s{start:06d}_n042",source_start=start,frames=42))
    index={r["id"]:r for r in virtual};sampler=Schedule(virtual,"robocasa",original["seed"])
    schedule=[dict(id=sampler.pick(i)[0]["id"]) for i in range(original["steps"]*original["global_batch"])]
    selections={"train":collections.defaultdict(set)}
    for entry in schedule:
        row=index[entry["id"]];selections["train"][row["episode_id"]].add(row["source_start"])
    for split in ["validation","test"]:
        selections[split]={}
        for row in rows[split]:
            digest=hashlib.sha256(f"{original['seed']}:{split}:{row['id']}".encode()).digest()
            rng=np.random.default_rng(int.from_bytes(digest[:8],"big"))
            selections[split][row["episode_id"]]={0 if episode_start else int(rng.integers(row["frames"]-42+1))}
    spec=dict(dataset="robocasa",method="memspo",reward="R0",initial_variant="sft",
        scope="Local CounterCubePreview success-only experiment; not original paper reproduction",
        upstream_commit=lock["commit"],source_lock_sha256=file_sha256(VENDOR/"SOURCE_LOCK.json"),
        original_recipe_sha256=file_sha256(VENDOR/"configs/train/robocasa_memspo.json"),
        steps=original["steps"],seed=original["seed"],global_batch=original["global_batch"],
        horizon=effective_horizon,world_size=a.gpus,bundle=str(Path(a.bundle).resolve()),
        text_model=str(Path(a.text_model).resolve()),bundle_manifest_sha256=file_sha256(Path(a.bundle)/"manifest.json"),
        source_manifests={k:file_sha256(v) for k,v in manifests.items()},
        window_frames=42,training_consumed_frames=effective_horizon+2,evaluation_consumed_frames=34,
        train_start_policy=("original episode first two frames; source_start=0 only; upstream episode sampler records fixed schedule" if episode_start else "all valid 42-frame starts; upstream task/episode/window sampler records fixed schedule"),
        evaluation_start_policy=("original episode first two frames; source_start=0 for every success episode" if episode_start else "one independent fixed seeded valid 42-frame start per episode; not centered"),
        possible_train_windows=len(virtual),scheduled_unique_windows=len({r["id"] for r in schedule}),
        scheduled_unique_episodes=len(selections["train"]),schedule_samples=len(schedule),
        success_episode_counts={k:len(v) for k,v in rows.items()},
        original_window_generation_unavailable=True,paper_locked=False,
        checkpoint_selection=("validation mean LPIPS32 over steps 50/100/150/200; earliest step breaks ties; test only selected checkpoint" if effective_horizon==32 else "final 200 only"))
    if episode_start:
        spec.update(initial_reference_policy="original_episode_first_two",initial_observations=2,gt_refresh=False)
    stable_json(out/"protocol.json",spec);stable_json(out/"schedule.json",schedule)
    audit={}
    for split in ["train","validation","test"]:
        folder=out/"data"/split;folder.mkdir(parents=True,exist_ok=True)
        starts=selections[split];jobs=[(manifests[split],r,starts[r["episode_id"]],folder) for r in rows[split] if r["episode_id"] in starts]
        results={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for i,cases in enumerate(pool.map(materialize,jobs),1):
                results.update({r["id"]:r for r in cases})
                if i%25==0 or i==len(jobs):print(json.dumps(dict(event="prepare",split=split,episodes_done=i,total=len(jobs))),flush=True)
        if split=="train":cases=[results[r["id"]] for r in virtual if r["id"] in results]
        else:cases=[results[f"{r['id']}_s{next(iter(starts[r['episode_id']])):06d}_n042"] for r in rows[split]]
        stable_json(folder/"manifest.json",dict(schema_version=1,dataset="robocasa",cases=cases,
            source_manifest_sha256=file_sha256(manifests[split]),success_only=True,
            action_alignment="actions[t]: frame t -> t+1; unused final action zero",
            selection=spec["train_start_policy"] if split=="train" else spec["evaluation_start_policy"]))
        audit[split]=dict(windows=len(cases),episodes=len(jobs),initial_states=len({r["initial_state_id"] for r in cases}),failure=0)
    cfg=dict(original)
    cfg["horizon"]=effective_horizon
    cfg.update(bundle=spec["bundle"],text_model=spec["text_model"],output=str(out/"train"),
        train_manifest=str(out/"data/train/manifest.json"),evaluation_manifest=str(out/"data/validation/manifest.json"),
        schedule=str(out/"schedule.json"),paper_locked=False)
    allowed={"bundle","text_model","output","train_manifest","evaluation_manifest","schedule","paper_locked"}
    assert all(cfg[k]==v for k,v in original.items() if k not in allowed | {"horizon"})
    stable_json(out/"train.json",cfg);stable_json(out/"data_audit.json",audit)
    comparison=dict(upstream_recipe=original,effective_recipe=cfg,
        unchanged_hyperparameters={k:v for k,v in original.items() if k not in allowed and cfg[k]==v},
        data_path_changes={k:dict(original=original.get(k),actual=cfg[k]) for k in allowed})
    if effective_horizon!=original["horizon"]:
        comparison["intentional_hyperparameter_changes"]={"horizon":dict(original=original["horizon"],actual=effective_horizon)}
    stable_json(out/"recipe_comparison.json",comparison)
    print(json.dumps(spec,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",nargs="?",default="all",choices=["prepare","preflight","baseline","train","evaluate","report","all"])
    for key in ["bundle","data-root","text-model","output"]:p.add_argument("--"+key,required=True)
    p.add_argument("--gpus",type=int,default=4);p.add_argument("--dry-run",action="store_true")
    a=p.parse_args();lock=verify_source()
    original=json.loads((VENDOR/"configs/train/robocasa_memspo.json").read_text());a.steps=original["steps"];a.seed=original["seed"]
    if a.gpus!=4:p.error("This launcher fixes 4 GPUs")
    if a.stage in ["prepare","all"]:prepare(a,lock)
    if a.stage in ["preflight","all"] and not a.dry_run:preflight(a)
    if a.stage in ["baseline","all"]:evaluate(a,"sft")
    if a.stage in ["train","all"]:train(a)
    if a.stage in ["evaluate","all"]:evaluate(a,"rl")
    if a.stage in ["report","all"] and not a.dry_run:
        report(a);f=Path(a.output)/"REPORT.md"
        f.write_text(f.read_text()+"\nReleased unmodified MemSPO trainer; original 200-step matched-ablation recipe. "
            "Custom success-only data, 42-frame windows, upstream task/episode/window sampler; RL consumes first 12 frames. "
            "Local window enumeration and fixed sequence are data adaptations, not exact reproduction of unavailable original window construction. "
            "Evaluation fixes one seeded 42-frame window per episode and consumes its first 34 frames; no centering. "
            "Final 200-step comparison only. Existing test split was used in earlier diagnosis; this is not a new blind test.\n")

if __name__=="__main__":main()
