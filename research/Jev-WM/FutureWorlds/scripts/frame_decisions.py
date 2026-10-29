"""One-timestamp multi-candidate Decisions pilot."""
import argparse,json,os
from pathlib import Path
import numpy as np
import focused_decisions as core
from futureworlds.frame_decisions import FrameStore,QUESTIONS,PROMPT,FORMULA
from futureworlds.focused_alignment import order_consensus
from futureworlds.data import write_json
from futureworlds.protocol import implementation
from futureworlds.checkpoints import file_sha256

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"runs/local_success_decisions"
EXP=PRIOR/"framewise"
BASE=ROOT/"runs/local_success_memspo_h32"
def load(p):return json.loads(Path(p).read_text())

def prepare():
    EXP.mkdir(parents=True,exist_ok=True)
    if (EXP/"protocol.json").exists():return load(EXP/"train.json")
    cfg=load(BASE/"train.json")
    manifest=load(cfg["train_manifest"]);index={r["id"]:r for r in manifest["cases"]}
    old=load(PRIOR/"protocol.json");focus=load(PRIOR/"focused/protocol.json")
    excluded={r["initial_state_id"] for r in old["calibration_rows"]+focus["initial_calibration_rows"]+focus["calibration_rows"]}
    schedule=load(PRIOR/"focused/schedule.json")
    excluded|={index[e["id"]]["initial_state_id"] for e in schedule}
    groups={}
    for row in manifest["cases"]:
        if row["initial_state_id"] not in excluded:groups.setdefault(row["initial_state_id"],[]).append(row)
    rng=np.random.default_rng(20261019)
    rows=[groups[g][int(rng.integers(len(groups[g])))] for g in rng.permutation(sorted(groups))[:3]]
    cfg.update(reward="decisions",steps=4,global_batch=2,checkpoint_every=1,output=str(EXP/"train"),
               schedule=str(EXP/"schedule.json"),evaluation_manifest=str(PRIOR/"validation/manifest.json"),
               decisions=dict(backend="api",model="gpt-6-luna",rubric="frame_decisions",
                              review_dir=str(EXP/"reviews"),max_api_calls=136,timeout_seconds=90,
                              retries=0,spacing_seconds=3))
    write_json(EXP/"train.json",cfg);write_json(EXP/"schedule.json",schedule)
    write_json(EXP/"protocol.json",dict(baseline=cfg["bundle"],baseline_sha256=file_sha256(Path(cfg["bundle"])/"manifest.json"),
                  original_two_frames=True,horizon=32,calibration_rows=rows,seed=20261019,
                  formula=FORMULA,prompt=PROMPT,questions=QUESTIONS,
                  gate=dict(minimum_controls_drop=.3,black_max=.1,clean_min=.8,
                            informative_groups=2,maximum_selected_lpips_ratio=1.05,
                            minimum_range=.04,maximum_order_delta=.15),
                  api_budget=dict(development=12,calibration=72,train=136),
                  sampling="8 fixed timestamps; one timestamp/reference/four predictions per API call; full+detail views, order and candidate-label reversal",
                  limitations="Small exploratory calibration; no blind test claim. LPIPS is image quality, not behavior correctness. Data from train success only.",
                  implementation=implementation()))
    return cfg

def configure_core():
    core.EXP=EXP;core.prepare=prepare

def store(max_calls):
    cfg=prepare();core.runtime(api=True)
    return FrameStore({**cfg["decisions"],"max_api_calls":max_calls})

def controls_for(z):
    controls,edited=core.corrupt(z["reference"])
    return np.stack([z["reference"],np.zeros_like(z["reference"]),controls["duplicate"],controls["missing"]])

def control_pass(result):
    values=np.asarray(result["frame_scores"])
    reference=values[:,:,0];black=values[:,:,1]
    gaps=reference[:,:,1,None]-values[:,:,2:,1]
    return bool(reference[:,:,:2].min()>=.8 and black.max()<=.1 and gaps.min()>=.3)

def develop():
    s=store(12)
    source=PRIOR/"calibration/g00045_b09_s000000_n042/candidates.npz"
    results={}
    with np.load(source) as z:
        for name,pred in [
            ("controls",controls_for(z)),
            ("candidates",z["prediction"]),
            ("freeze",np.stack([z["reference"],np.repeat(z["initial"][-1:],32,axis=0),z["prediction"][0],z["prediction"][1]]))]:
            results[name]=s.group(z["initial"],pred,z["actions"][2:],z["reference"],str(z["instruction"].item()),[8,13])
            write_json(EXP/f"development_{name}.json",results[name])
    passed=control_pass(results["controls"])
    freeze=np.asarray(results["freeze"]["frame_scores"])
    passed &= bool((freeze[:,:,0,0]-freeze[:,:,1,0]).min()>=.1)
    controls_passed=bool(passed)
    passed &= not results["candidates"]["consensus"]["rejected"]
    report=dict(state="PASS" if passed else "DEVELOPMENT_FAILED",controls_passed=controls_passed,new_api_calls=s.client.calls,
                results=results,implementation=implementation())
    write_json(EXP/"development.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="results"}),flush=True)
    return passed

def collect():
    configure_core();core.collect()

def calibrate():
    if load(EXP/"development.json")["state"]!="PASS":raise RuntimeError("Development failed")
    s=store(72);protocol=load(EXP/"protocol.json");results=[]
    for row in protocol["calibration_rows"]:
        folder=EXP/"calibration"/row["id"];meta=load(folder/"metadata.json")
        assert meta["sha256"]==file_sha256(folder/"candidates.npz")
        assert meta["implementation"]==implementation()
        with np.load(folder/"candidates.npz") as z:
            # Controls at two spaced observed timestamps, candidates at all8 judged times.
            controls=s.group(z["initial"],controls_for(z),z["actions"][2:],z["reference"],str(z["instruction"].item()),[8,22])
            result=s.group(z["initial"],z["prediction"],z["actions"][2:],z["reference"],str(z["instruction"].item()))
        write_json(folder/"controls.json",controls);write_json(folder/"scores.json",result)
        best=int(np.argmax(result["rewards"]));lp=[m["lpips"] for m in meta["metrics"]]
        results.append(dict(id=row["id"],controls_pass=control_pass(controls),reliable=not result["consensus"]["rejected"],
                            best_candidate=best,selected_lpips=lp[best],mean_lpips=float(np.mean(lp)),metrics=meta["metrics"],scores=result))
        print(json.dumps(dict(event="case_calibrated",id=row["id"],controls=results[-1]["controls_pass"],consensus=result["consensus"])),flush=True)
    reliable=[r for r in results if r["reliable"]]
    ratio=sum(r["selected_lpips"] for r in reliable)/sum(r["mean_lpips"] for r in reliable) if reliable else None
    passed=all(r["controls_pass"] for r in results) and len(reliable)>=2 and ratio<=1.05
    report=dict(state="PASS" if passed else "CALIBRATION_FAILED",results=results,reliable_groups=len(reliable),
                selected_to_mean_lpips_ratio=ratio,new_api_calls=s.client.calls,implementation=implementation())
    write_json(EXP/"calibration.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="results"}),flush=True)
    return passed

def train(resume=False):
    configure_core();core.train_run(resume)

def evaluate():
    configure_core();core.evaluate_run()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","develop","collect","calibrate","train","evaluate","all"]);p.add_argument("--resume",action="store_true");a=p.parse_args()
    prepare()
    try:
        if a.stage=="all":
            if develop():
                collect()
                if calibrate():train(a.resume);evaluate()
        elif a.stage=="train":train(a.resume)
        else:globals()[a.stage]()
    except Exception as exc:
        write_json(EXP/"status.json",dict(state="STOPPED",stage=a.stage,error_type=type(exc).__name__,error=str(exc)))
        raise

