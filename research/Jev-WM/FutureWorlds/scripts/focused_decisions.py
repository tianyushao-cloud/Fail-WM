"""Focused pure Decisions pilot, server-only and independent of the four-GPU R0 run."""
import argparse, gc, json, os, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from scipy import ndimage

from futureworlds.checkpoints import file_sha256
from futureworlds.data import read_case,read_manifest,write_json
from futureworlds.protocol import implementation
from futureworlds.focused_alignment import FocusedStore,QUESTIONS,PROMPT,FORMULA,red_mask,crop_box,rgb,order_consensus

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"runs/local_success_decisions"
EXP=PRIOR/"focused"
BASE=ROOT/"runs/local_success_memspo_h32"

def load(p):return json.loads(Path(p).read_text())

def runtime(api=False):
    torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.25,0)
    if api:
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"]=Path("/root/.config/jev-wm/openai.key").read_text().strip()
        for name in ("HTTPS_PROXY","HTTP_PROXY","https_proxy","http_proxy"):
            os.environ[name]="http://127.0.0.1:17897"
        os.environ["NO_PROXY"]=os.environ["no_proxy"]="127.0.0.1,localhost"

def prepare():
    EXP.mkdir(parents=True,exist_ok=True)
    if (EXP/"protocol.json").exists():return load(EXP/"train.json")
    cfg=load(BASE/"train.json")
    manifest=read_manifest(cfg["train_manifest"],"robocasa")
    assert all(r["success"] and r["source_start"]==0 for r in manifest["cases"])
    old=load(PRIOR/"protocol.json")
    excluded={r["initial_state_id"] for r in old["calibration_rows"]}
    groups={}
    for row in manifest["cases"]:
        if row["initial_state_id"] not in excluded:
            groups.setdefault(row["initial_state_id"],[]).append(row)
    rng=np.random.default_rng(20261016)
    group_ids=rng.permutation(sorted(groups))[:3]
    rows=[groups[g][int(rng.integers(len(groups[g])))] for g in group_ids]
    excluded|=set(group_ids)
    index={r["id"]:r for r in manifest["cases"]}
    schedule=[e for e in load(BASE/"schedule.json") if index[e["id"]]["initial_state_id"] not in excluded][:8]
    cfg.update(reward="decisions",steps=4,global_batch=2,checkpoint_every=1,
               schedule=str(EXP/"schedule.json"),output=str(EXP/"train"),
               evaluation_manifest=str(PRIOR/"validation/manifest.json"),
               decisions=dict(backend="api",model="gpt-6-luna",rubric="focused_alignment",
                              review_dir=str(EXP/"reviews"),max_api_calls=72,timeout_seconds=90,
                              retries=0,spacing_seconds=6,minimum_range=.04,maximum_order_delta=.15))
    write_json(EXP/"train.json",cfg);write_json(EXP/"schedule.json",schedule)
    write_json(EXP/"protocol.json",dict(
        source_bundle=cfg["bundle"],source_bundle_sha256=file_sha256(Path(cfg["bundle"])/"manifest.json"),
        baseline="443-success SFT step10000 + success codec step44000; original frames0/1; horizon32",
        reward=FORMULA,questions=QUESTIONS,prompt=PROMPT,
        full_frames=8,detail_frames=16,crop_rule="reference red-cube extent with fixed padding shared across candidates",
        calibration_rows=rows,development_id="g00045_b09_s000000_n042",
        training_steps=4,global_batch=2,validation_manifest=cfg["evaluation_manifest"],
        selected_checkpoint="last step4, no validation checkpoint selection",
        max_attempts=dict(development=16,calibration=56,training=72),
        gate=dict(clean_min=.8,black_max=.10,freeze_gap=.1,object_corruption_gap=.15,
                  maximum_order_delta=.15,minimum_candidate_range=.04,informative_groups=2,
                  maximum_selected_lpips_ratio=1.05),
        limitations="Three new calibration groups, training split only; not an untouched test. Pixel metric agreement is not semantic correctness. Four previously inspected validation groups. No claim of improvement without actual SFT/RL comparison.",
        implementation=implementation()))
    return cfg

def corrupt(reference):
    # Dataset-specific controls: edit only red-cube pixels; unrelated scene stays unchanged.
    missing=reference.copy();duplicate=reference.copy();edited=[]
    for t,frame in enumerate(reference):
        mask=red_mask(frame)
        labels,n=ndimage.label(mask)
        if not n:continue
        counts=np.bincount(labels.ravel());counts[0]=0
        component=labels==counts.argmax()
        if component.sum()<8:continue
        yy,xx=np.where(component)
        expanded=ndimage.binary_dilation(component,iterations=2)
        nearest=ndimage.distance_transform_edt(expanded,return_distances=False,return_indices=True)
        missing[t][expanded]=frame[nearest[0][expanded],nearest[1][expanded]]
        dy,dx=-28,-32
        valid=(yy+dy>=0)&(xx+dx>=0)
        duplicate[t,yy[valid]+dy,xx[valid]+dx]=frame[yy[valid],xx[valid]]
        edited.append(t)
    if len(edited)<8:raise ValueError("Insufficient visible red cube for controlled corruptions")
    return dict(missing=missing,duplicate=duplicate),edited

def review_sheet(folder,z,options):
    indices=[0,4,8,13,17,22,26,31]
    canvas=Image.new("RGB",(1280,len(options)*148),"white");draw=ImageDraw.Draw(canvas)
    for row,(label,frames) in enumerate(options.items()):
        for col,t in enumerate(indices):
            canvas.paste(Image.fromarray(frames[t]).resize((160,128)),(col*160,row*148+20))
            draw.text((col*160,row*148),f"{label} t={t+2}",fill="black")
    canvas.save(folder/"review.jpg")
    box=crop_box(z["initial"],z["reference"])
    canvas=Image.new("RGB",(1280,len(options)*180),"white");draw=ImageDraw.Draw(canvas)
    for row,(label,frames) in enumerate(options.items()):
        for col,t in enumerate(indices):
            canvas.paste(Image.fromarray(frames[t]).crop(box).resize((160,160)),(col*160,row*180+20))
            draw.text((col*160,row*180),f"{label} t={t+2}",fill="black")
    canvas.save(folder/"details.jpg")

def options_for(z,development=False):
    corrupted,indices=corrupt(z["reference"])
    options=dict(reference=z["reference"],black=np.zeros_like(z["reference"]),
                 frozen=np.repeat(z["initial"][-1:],32,axis=0),**corrupted)
    if development:
        options.update(candidate_0=z["prediction"][0],candidate_1=z["prediction"][1])
    else:
        options.update({f"candidate_{i}":p for i,p in enumerate(z["prediction"])})
    return options,indices

def dev_images():
    prepare()
    source=PRIOR/"calibration/g00045_b09_s000000_n042/candidates.npz"
    folder=EXP/"development";folder.mkdir(exist_ok=True)
    with np.load(source,allow_pickle=False) as z:
        options,indices=options_for(z,True)
        np.savez_compressed(folder/"controls.npz",**options)
        review_sheet(folder,z,options)
    write_json(folder/"controls.json",dict(source=str(source),sha256=file_sha256(source),edited_offsets=indices))
    print("Development controls rendered",flush=True)

def control_check(scores,gate):
    # Each presentation order must pass separately.
    rows=[]
    for side in [0,1]:
        value={k:v["orders"][side]["score"] for k,v in scores.items()}
        components={k:v["orders"][side]["components"] for k,v in scores.items()}
        passed=(value["reference"]>=gate["clean_min"] and value["black"]<=gate["black_max"]
                and value["reference"]-value["frozen"]>=gate["freeze_gap"]
                and all(components["reference"]["object_behavior"]-components[k]["object_behavior"]>=gate["object_corruption_gap"] for k in ("duplicate","missing")))
        rows.append(dict(prediction_first=bool(side),passed=bool(passed),scores=value,
                         corruption_object_gaps={k:components["reference"]["object_behavior"]-components[k]["object_behavior"] for k in ("duplicate","missing")}))
    return dict(passed=all(r["passed"] for r in rows),orders=rows)

def score_case(store,source,folder,development=False):
    folder.mkdir(parents=True,exist_ok=True)
    results=load(folder/"scores.json") if (folder/"scores.json").exists() else {}
    with np.load(source,allow_pickle=False) as z:
        options,edited=options_for(z,development)
        review_sheet(folder,z,options)
        for label,pred in options.items():
            orders=[]
            for reverse in [False,True]:
                store.prediction_first=reverse
                key,value=store.score_arrays(z["initial"],pred,z["actions"][2:],str(z["instruction"].item()),"robocasa",z["reference"])
                orders.append(dict(key=key,score=value,components=store.components(key)))
                print(json.dumps(dict(case=folder.name,label=label,prediction_first=reverse,score=value,api_calls=store.client.calls)),flush=True)
            results[label]=dict(orders=orders,mean=float(np.mean([s["score"] for s in orders])),
                                order_delta=abs(orders[0]["score"]-orders[1]["score"]))
            write_json(folder/"scores.json",results)
    return results

def develop():
    cfg=prepare();runtime(api=True)
    store=FocusedStore({**cfg["decisions"],"max_api_calls":16})
    scores=score_case(store,PRIOR/"calibration/g00045_b09_s000000_n042/candidates.npz",EXP/"development",True)
    result=control_check(scores,load(EXP/"protocol.json")["gate"])
    result.update(state="PASS" if result["passed"] else "DEVELOPMENT_FAILED",new_api_calls=store.client.calls)
    write_json(EXP/"development.json",result);print(json.dumps(result),flush=True)
    return result["passed"]

def collect():
    cfg=prepare();runtime()
    from futureworlds.pipeline import WorldPipeline
    from futureworlds.search import rollout_group_beam
    from futureworlds.rewards import CodecWorker
    from futureworlds.batch import TensorBatch
    from futureworlds.evaluation import Metrics
    pipe=WorldPipeline(cfg["bundle"],"robocasa","sft","cuda",cfg["text_model"])
    worker=CodecWorker(pipe.codec,None);metric=Metrics("cuda")
    torch.manual_seed(cfg["seed"])
    for row in load(EXP/"protocol.json")["calibration_rows"]:
        folder=EXP/"calibration"/row["id"]
        if (folder/"metadata.json").exists():
            meta=load(folder/"metadata.json")
            assert file_sha256(folder/"candidates.npz")==meta["sha256"]
            assert meta["implementation"]==implementation()
            continue
        started=time.monotonic()
        images,actions,task=read_case(cfg["train_manifest"],row,pipe.action_dim,32)
        with torch.no_grad():
            ctx,dyn,acts,pixels,cond=pipe.encode_training_window(images,actions,task)
            frames,trace=rollout_group_beam(pipe.model,ctx.reshape(1,-1)+4375,dyn[1:2],acts,32,penalty=.1)
            codes=torch.stack([f["response"] for f in frames],1)
            predictions=worker.detokenize(TensorBatch(dict(tokens=codes,ctx_tokens=ctx.expand(len(codes),-1,-1))),None).batch["pixels"][:,1:]
            pred=np.stack([rgb(p) for p in predictions])
            metrics=[metric(p.astype(np.float32)/255,images[2:],[32])["metrics"]["32"] for p in pred]
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/"candidates.npz",initial=images[:2],reference=images[2:],prediction=pred,actions=actions,instruction=np.array(task))
        write_json(folder/"metadata.json",dict(id=row["id"],sha256=file_sha256(folder/"candidates.npz"),
                   implementation=implementation(),metrics=metrics,trace=trace,seconds=time.monotonic()-started))
        with np.load(folder/"candidates.npz") as z:
            options,_=options_for(z);review_sheet(folder,z,options)
        print(json.dumps(dict(event="candidates_ready",id=row["id"],seconds=time.monotonic()-started)),flush=True)
    del pipe,worker,metric
    gc.collect();torch.cuda.empty_cache()

def calibrate():
    cfg=prepare();runtime(api=True)
    if load(EXP/"development.json")["state"]!="PASS":raise RuntimeError("Development controls failed")
    store=FocusedStore({**cfg["decisions"],"max_api_calls":56})
    protocol=load(EXP/"protocol.json");gate=protocol["gate"];results=[]
    for row in protocol["calibration_rows"]:
        folder=EXP/"calibration"/row["id"];meta=load(folder/"metadata.json")
        assert meta["implementation"]==implementation()
        assert meta["sha256"]==file_sha256(folder/"candidates.npz")
        scores=score_case(store,folder/"candidates.npz",folder)
        controls=control_check(scores,gate)
        candidates=[scores[f"candidate_{i}"] for i in range(4)]
        means=np.array([c["mean"] for c in candidates])
        order0=np.array([c["orders"][0]["score"] for c in candidates])
        order1=np.array([c["orders"][1]["score"] for c in candidates])
        means,consensus=order_consensus([order0,order1],gate["minimum_candidate_range"],gate["maximum_order_delta"])
        flips=consensus["clear_flips"]
        reliable=not consensus["rejected"]
        best=int(means.argmax());lp=[m["lpips"] for m in meta["metrics"]]
        results.append(dict(id=row["id"],controls=controls,reliable=bool(reliable),
                            best_candidate=best,selected_lpips=lp[best],mean_lpips=float(np.mean(lp)),
                            order_flips=flips,ambiguous_pairs=consensus["ambiguous_pairs"],score_range=float(np.ptp(means)),scores=scores,metrics=meta["metrics"]))
    reliable=[r for r in results if r["reliable"]]
    ratio=sum(r["selected_lpips"] for r in reliable)/sum(r["mean_lpips"] for r in reliable) if reliable else None
    passed=(all(r["controls"]["passed"] for r in results) and len(reliable)>=gate["informative_groups"]
            and ratio<=gate["maximum_selected_lpips_ratio"])
    report=dict(state="PASS" if passed else "CALIBRATION_FAILED",results=results,
                reliable_groups=len(reliable),selected_to_mean_lpips_ratio=ratio,
                new_api_calls=store.client.calls,gate=gate,implementation=implementation())
    write_json(EXP/"calibration.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("results","gate","implementation")}),flush=True)
    return passed

def train_run(resume=False):
    cfg=prepare();runtime(api=True)
    audit=load(EXP/"calibration.json")
    if audit["state"]!="PASS" or audit["implementation"]!=implementation():
        raise RuntimeError("No passing calibration for this implementation")
    from futureworlds.training import train
    path=EXP/"train.json"
    if resume:
        cfg["resume"]=str(sorted((EXP/"train/checkpoints").glob("step_*"))[-1])
        path=EXP/"resume.json";write_json(path,cfg)
    train(path)

def evaluate_run():
    cfg=prepare();runtime()
    from futureworlds.exporting import export_model
    from futureworlds.evaluation import evaluate
    if not (EXP/"train/complete.json").exists():raise RuntimeError("Training incomplete")
    bundle=EXP/"rl_bundle";export_model(cfg["bundle"],"robocasa",EXP/"train/checkpoints/step_000004",bundle)
    for label,source,variant in [("sft",cfg["bundle"],"sft"),("rl",bundle,"main")]:
        evaluate(str(source),"robocasa",variant,cfg["evaluation_manifest"],str(EXP/"evaluation"/label),
                 device="cuda",metric_device="cuda",text_model=cfg["text_model"],horizons=[10,20,32],beam_width=4)
        gc.collect();torch.cuda.empty_cache()
    sft=load(EXP/"evaluation/sft/summary.json");rl=load(EXP/"evaluation/rl/summary.json")
    comparison={h:{k:dict(sft=v,rl=rl["aggregate"][h][k],delta=rl["aggregate"][h][k]-v)
                  for k,v in values.items()} for h,values in sft["aggregate"].items()}
    write_json(EXP/"comparison.json",comparison)
    print(json.dumps(comparison),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["prepare","dev_images","develop","collect","calibrate","train","evaluate","all"])
    parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    prepare()
    try:
        if args.stage=="all":
            if develop():
                collect()
                if calibrate():
                    train_run(args.resume);evaluate_run()
        elif args.stage=="train":train_run(args.resume)
        else:globals()[args.stage if args.stage!="evaluate" else "evaluate_run"]()
    except Exception as exc:
        write_json(EXP/"status.json",dict(state="STOPPED",stage=args.stage,error_type=type(exc).__name__,error=str(exc)))
        raise

