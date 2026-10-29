"""Success-only MemSPO experiment runner. All outputs stay under --output."""
import argparse, collections, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from futureworlds.checkpoints import file_sha256, verify_bundle
from futureworlds.data import read_manifest, read_case, write_json

ROOT = Path(__file__).resolve().parents[1]

def stable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Configuration changed: {path}. Use a new output directory.")
    else:
        write_json(path, value)

def prepare(a):
    output=Path(a.output).resolve(); source=Path(a.data_root).resolve()
    output.mkdir(parents=True,exist_ok=True)
    spec=dict(dataset="robocasa", method="memspo", reward="R0", reward_formula="-mean_t(MAE + LPIPS_VGG)",
              initial_variant="sft", steps=a.steps, global_batch=16, horizon=10, seed=a.seed,
              world_size=a.gpus, evaluation_horizons=[10,20,32], beam_width=4,
              train_window="centered 12 frames", evaluation_window="centered 34 frames",
              checkpoint_selection="final step only; no validation-based selection",
              scope="local CounterCubePreview success-only experiment; not paper 128-case reproduction",
              bundle=str(Path(a.bundle).resolve()),text_model=str(Path(a.text_model).resolve()),
              source_manifests={n:file_sha256(source/f) for n,f in
                                [("train","train_success.json"),("validation","val.json"),("test","test.json")]},
              bundle_manifest_sha256=file_sha256(Path(a.bundle)/"manifest.json"))
    stable_json(output/"protocol.json",spec)
    selected={}; sets={}
    for split,name,length in [("train","train_success.json",12),("validation","val.json",34),("test","test.json",34)]:
        manifest=source/name; original=read_manifest(manifest,"robocasa")
        rows=[r for r in original["cases"] if r.get("success") is True]
        if not rows: raise ValueError(f"No success trajectories: {manifest}")
        if split=="train" and len(rows)!=len(original["cases"]):raise ValueError("train_success contains failures")
        if any(r["frames"]<length for r in rows):raise ValueError(f"Short success episode in {split}; do not silently drop")
        group_ids=[r.get("initial_state_id") for r in rows]
        if any(v is None for v in group_ids):raise ValueError("Missing initial_state_id")
        sets[split]=dict(groups=set(group_ids),episodes={r["episode_id"] for r in rows})
        folder=output/"data"/split;folder.mkdir(parents=True,exist_ok=True);cases=[]
        for i,r in enumerate(rows):
            images,actions,text=read_case(manifest,r,52)
            start=(len(images)-length)//2; target=folder/f"{i:06d}.npz"
            values=dict(images=images[start:start+length],actions=actions[start:start+length],instruction=np.array(text))
            if target.exists():
                with np.load(target,allow_pickle=False) as z:
                    if any(not np.array_equal(z[k],v) for k,v in values.items()):raise ValueError(f"Changed data: {target}")
            else:
                with target.with_suffix(".partial").open("wb") as stream:np.savez_compressed(stream,**values)
                target.with_suffix(".partial").replace(target)
            cases.append(dict(id=r["id"],episode_id=r["episode_id"],initial_state_id=r["initial_state_id"],
                              file=target.name,sha256=file_sha256(target),frames=length,split=r["split"],
                              task=r["task"],success=True,source_manifest=str(manifest),source_file=r["file"],
                              source_sha256=r["sha256"],source_start=start))
        stable_json(folder/"manifest.json",dict(schema_version=1,dataset="robocasa",cases=cases,
                    selection="all success=True episodes in original order; centered fixed windows",
                    source_manifest_sha256=file_sha256(manifest)))
        selected[split]=cases
    for i,x in enumerate(sets):
        for y in list(sets)[i+1:]:
            for k in ["groups","episodes"]:
                if sets[x][k]&sets[y][k]:raise ValueError(f"Split leakage: {x}/{y}/{k}")
    schedule=[];rng=np.random.default_rng(a.seed)
    while len(schedule)<a.steps*16:
        schedule.extend(dict(id=selected["train"][i]["id"]) for i in rng.permutation(len(selected["train"])))
    stable_json(output/"schedule.json",schedule[:a.steps*16])
    cfg=json.loads((ROOT/"configs/train/robocasa_memspo.json").read_text())
    cfg.update(bundle=spec["bundle"],text_model=spec["text_model"],steps=a.steps,seed=a.seed,
               train_manifest=str(output/"data/train/manifest.json"),
               evaluation_manifest=str(output/"data/validation/manifest.json"),
               output=str(output/"train"),schedule=str(output/"schedule.json"),paper_locked=False,checkpoint_every=50)
    stable_json(output/"train.json",cfg)
    summary={k:dict(cases=len(v),initial_states=len(sets[k]["groups"]),success=len(v),failure=0) for k,v in selected.items()}
    stable_json(output/"data_audit.json",summary);print(json.dumps(summary,indent=2),flush=True)

def launch(a,command,args,distributed=True):
    cmd=[sys.executable]
    if distributed and a.gpus>1:cmd+=["-m","torch.distributed.run","--standalone",f"--nproc_per_node={a.gpus}"]
    cmd+=["-m","futureworlds",command]+[str(v) for v in args]
    print(json.dumps({"command":cmd}),flush=True)
    if not a.dry_run:subprocess.run(cmd,check=True,env=dict(os.environ,USE_TF="0",USE_FLAX="0",TOKENIZERS_PARALLELISM="false"))

def train(a):
    out=Path(a.output).resolve();cfg=json.loads((out/"train.json").read_text());path=out/"train.json"
    complete=out/"train/complete.json"
    if complete.exists():
        if json.loads(complete.read_text())["steps"]!=a.steps:raise ValueError("Completed run step mismatch")
        print("Training already complete",flush=True);return
    checkpoints=sorted((out/"train/checkpoints").glob("step_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if checkpoints:
        cfg["resume"]=str(checkpoints[-1]);path=out/"resume.json";write_json(path,cfg)
    elif (out/"train/run.json").exists():
        raise RuntimeError("Interrupted before first checkpoint. Keep diagnostics and choose a new output; cannot resume without optimizer state.")
    launch(a,"train",["--config",path])

def evaluate(a,arm):
    out=Path(a.output).resolve()
    if arm=="sft":bundle=Path(a.bundle).resolve();variant="sft"
    else:
        checkpoint=out/"train/checkpoints"/f"step_{a.steps:06d}";bundle=out/"rl_bundle";variant="main"
        if not a.dry_run and not (checkpoint/"experiment.json").exists():raise ValueError("Final RL checkpoint missing")
        launch(a,"export-model",["--bundle",a.bundle,"--dataset","robocasa","--checkpoint",checkpoint,"--output",bundle],False)
    for split in ["validation","test"]:
        launch(a,"evaluate",["--bundle",bundle,"--dataset","robocasa","--variant",variant,
               "--manifest",out/"data"/split/"manifest.json","--output",out/"evaluation"/split/arm,
               "--device","cuda","--metric-device","cuda","--text-model",a.text_model,
               "--horizons",10,20,32,"--beam-width",4,"--memory","full"])

def report(a):
    out=Path(a.output).resolve();results={}
    cfg=json.loads((out/"train.json").read_text())
    selection=json.loads((out/"checkpoint_selection.json").read_text()) if (out/"checkpoint_selection.json").exists() else None
    checkpoint_description=(f"按验证集32帧LPIPS选择第{selection['selected_step']}步" if selection else "固定最后一步")
    lines=["# 成功轨迹 MemSPO 与原始 SFT 对照","",
           "本实验使用本地 CounterCubePreview 成功轨迹，不是论文固定 128-case 复现。训练与验证/测试按初始状态分离。",
           f"训练：原始 SFT 初始化，MemSPO + R0，{a.steps} 步，全局 batch=16，horizon={cfg['horizon']}，lr=1e-6，KL=0.001。R0 为负的 MAE 与 LPIPS 之和，不调用 Decisions。",
           f"评估：{checkpoint_description}，Beam4，FP32，两个初始观测，不注入未来真实帧。主要指标为独立测试集 32 帧 LPIPS；同时报告 10/20/32 帧各指标。",
           "","| 划分 | 帧数 | 指标 | SFT | RL | RL−SFT |","|---|---:|---|---:|---:|---:|"]
    for split in ["validation","test"]:
        s=json.loads((out/"evaluation"/split/"sft/summary.json").read_text())
        r=json.loads((out/"evaluation"/split/"rl/summary.json").read_text())
        for key in ["manifest_sha256","cases","horizons","beam_width","memory","text_assets","implementation","precision","gt_refresh"]:
            assert s["protocol"][key]==r["protocol"][key],key
        assert s["state"]==r["state"]=="COMPLETE"
        manifest=json.loads((out/"data"/split/"manifest.json").read_text())
        groups=collections.defaultdict(list)
        for i,row in enumerate(manifest["cases"]):groups[row["initial_state_id"]].append(i)
        results[split]={}
        for h in ["10","20","32"]:
            results[split][h]={}
            for k in ["lpips","mae","psnr","ssim"]:
                sv=s["aggregate"][h][k];rv=r["aggregate"][h][k]
                delta=np.array([rr["metrics"][h][k]-ss["metrics"][h][k] for ss,rr in zip(s["per_case"],r["per_case"])])
                rng=np.random.default_rng(a.seed);blocks=list(groups.values());sums=np.array([delta[b].sum() for b in blocks]);counts=np.array([len(b) for b in blocks])
                draws=rng.integers(len(blocks),size=(10000,len(blocks)));means=sums[draws].sum(1)/counts[draws].sum(1)
                ci=np.quantile(means,[.025,.975]).tolist()
                results[split][h][k]=dict(sft=sv,rl=rv,delta=rv-sv,cluster_bootstrap95=ci,
                    improved_cases=int(np.sum(delta<0 if k in ["lpips","mae"] else delta>0)),cases=len(delta),initial_states=len(groups))
                lines.append(f"| {split} | {h} | {k} | {sv:.6f} | {rv:.6f} | {rv-sv:+.6f} |")
    primary=results["test"]["32"]["lpips"];ci=primary["cluster_bootstrap95"]
    conclusion=("测试集 32 帧 LPIPS 均值改善" if primary["delta"]<0 else "测试集 32 帧 LPIPS 均值未改善")
    lines+=["",conclusion+f"；RL−SFT={primary['delta']:+.6f}，按初始状态分组 bootstrap 95% 区间={ci}。",
            "单种子、小规模任务；其余指标为描述性结果。区间跨零时不能宣称稳健提升。",
            ("Checkpoint按验证集32帧LPIPS选择；测试集只评估所选权重。" if selection else "没有按验证分数挑选 checkpoint。")+"RL 的预测指标改善不等于机器人任务成功率提升。"]
    write_json(out/"comparison.json",results);(out/"REPORT.md").write_text("\n".join(lines)+"\n")

def preflight(a):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<a.gpus:
        raise ValueError(f"Need {a.gpus} CUDA devices, visible={torch.cuda.device_count()}, torch={torch.__version__}")
    verify_bundle(a.bundle)
    print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,visible_gpus=torch.cuda.device_count())),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","preflight","baseline","train","evaluate","report","all"])
    p.add_argument("--bundle",required=True);p.add_argument("--data-root",required=True);p.add_argument("--text-model",required=True)
    p.add_argument("--output",required=True);p.add_argument("--gpus",type=int,default=8);p.add_argument("--steps",type=int,default=200)
    p.add_argument("--seed",type=int,default=20261013);p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if a.gpus not in [1,2,4,8,16] or a.steps<1:p.error("gpus must divide global batch 16; steps must be positive")
    if a.stage in ["prepare","all"]:prepare(a)
    if a.stage in ["preflight","all"] and not a.dry_run:preflight(a)
    if a.stage in ["baseline","all"]:evaluate(a,"sft")
    if a.stage in ["train","all"]:train(a)
    if a.stage in ["evaluate","all"]:evaluate(a,"rl")
    if a.stage in ["report","all"] and not a.dry_run:report(a)
if __name__=="__main__":main()
