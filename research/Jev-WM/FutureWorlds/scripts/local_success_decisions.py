"""Bounded Decisions calibration and four-update pilot on the adapted success SFT."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from futureworlds.data import read_case, read_manifest, write_json
from futureworlds.checkpoints import file_sha256
from futureworlds.reference_alignment import ReferenceAlignmentStore, QUESTIONS, PROMPT, rgb
from futureworlds.protocol import implementation

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "runs/local_success_decisions"
BASE = ROOT / "runs/local_success_memspo_h32"


def load(p):
    return json.loads(Path(p).read_text())


def prepare():
    if (EXP / "protocol.json").exists():
        return load(EXP / "train.json")
    cfg = load(BASE / "train.json")
    manifest = read_manifest(cfg["train_manifest"], "robocasa")
    assert all(r["success"] and r["source_start"] == 0 for r in manifest["cases"])
    rng = np.random.default_rng(20261015)
    groups = {}
    for row in manifest["cases"]:
        groups.setdefault(row["initial_state_id"], []).append(row)
    selected = rng.permutation(sorted(groups))[:4]
    calibration = [groups[g][int(rng.integers(len(groups[g])))] for g in selected]
    schedule = [e for e in load(BASE / "schedule.json")
                if next(r for r in manifest["cases"] if r["id"] == e["id"])["initial_state_id"] not in selected][:8]
    cfg.update(reward="decisions", steps=4, global_batch=2, checkpoint_every=1,
               output=str(EXP / "train"), schedule=str(EXP / "schedule.json"),
               decisions=dict(backend="api", model="gpt-6-luna", rubric="reference_alignment",
                              review_dir=str(EXP / "reviews"), max_api_calls=40,
                              timeout_seconds=90, retries=0, spacing_seconds=6, minimum_range=.03))
    val_source = Path(cfg["evaluation_manifest"])
    val = read_manifest(val_source, "robocasa")
    val_groups = {}
    for row in val["cases"]:
        val_groups.setdefault(row["initial_state_id"], []).append(row)
    val_rows = [val_groups[g][0] for g in rng.permutation(sorted(val_groups))[:4]]
    dest = EXP / "validation"
    dest.mkdir(parents=True, exist_ok=True)
    for row in val_rows:
        os.link(val_source.parent / row["file"], dest / row["file"])
    write_json(dest / "manifest.json", {**val, "cases": val_rows})
    cfg["evaluation_manifest"] = str(dest / "manifest.json")
    write_json(EXP / "schedule.json", schedule)
    write_json(EXP / "train.json", cfg)
    write_json(EXP / "protocol.json", dict(
        baseline="443-success SFT step10000 + paired success codec step44000",
        source_bundle=cfg["bundle"], source_bundle_sha256=file_sha256(Path(cfg["bundle"]) / "manifest.json"),
        train_manifest_sha256=file_sha256(cfg["train_manifest"]),
        calibration_rows=calibration, validation_ids=[r["id"] for r in val_rows],
        prediction_horizon=32, initial_observations="original episode frames 0 and 1",
        reward="mean of three normalized official Decisions scores; no reconstruction term",
        reference_future="reviewer only; actor receives no future ground-truth observations",
        sampled_offsets=[0,4,8,13,17,22,26,31], questions=QUESTIONS, prompt=PROMPT,
        calibration_api_attempt_budget=32, training_api_attempt_budget=40,
        gate=dict(reference_over_freeze_min=.05, reference_over_black_min=.20,
                  informative_candidate_range=.03, informative_groups_min=2,
                  decisive_metric_relative_gap=.05, decisive_reward_gap=.03,
                  decisive_pairs_min=3, minimum_lpips_direction_fraction=2/3),
        stop_rule="Stop if calibration gate fails or any API/training error; never synthesize scores. Four updates only, compare last checkpoint with fixed SFT.",
        validation_limit="Four previously inspected validation groups; pilot only, not a new blind test. Never select checkpoint by these results.",
        training_framework="Existing FutureWorlds Decisions fork, MemSPO 4 groups x 2 beams, LR1e-6, KL.001, clip.2",
        implementation=implementation()))
    return cfg


def runtime(api=False):
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    if api:
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = Path("/root/.config/jev-wm/openai.key").read_text().strip()
        # Uses the already verified SSH reverse forwarding to the user's Clash.
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[name] = os.environ.get(name) or "http://127.0.0.1:17897"
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"


def collect():
    from futureworlds.pipeline import WorldPipeline
    from futureworlds.search import rollout_group_beam
    from futureworlds.rewards import CodecWorker
    from futureworlds.batch import TensorBatch
    from futureworlds.evaluation import Metrics
    cfg = prepare()
    runtime()
    torch.manual_seed(cfg["seed"])
    pipe = WorldPipeline(cfg["bundle"], "robocasa", "sft", "cuda", cfg["text_model"])
    worker = CodecWorker(pipe.codec, None)
    metric = Metrics("cuda")
    for row in load(EXP / "protocol.json")["calibration_rows"]:
        folder = EXP / "calibration" / row["id"]
        if (folder / "metadata.json").exists():
            saved=load(folder / "metadata.json")
            assert file_sha256(folder / "candidates.npz") == saved["sha256"]
            assert saved["implementation"] == implementation()
            continue
        started=time.monotonic()
        images, actions, task = read_case(cfg["train_manifest"], row, pipe.action_dim, 32)
        with torch.no_grad():
            ctx, dyn, acts, pixels, cond = pipe.encode_training_window(images, actions, task)
            frames, trace = rollout_group_beam(pipe.model, ctx.reshape(1,-1)+4375, dyn[1:2], acts, 32, penalty=.1)
            codes = torch.stack([f["response"] for f in frames], 1)
            predictions = worker.detokenize(TensorBatch(dict(tokens=codes,ctx_tokens=ctx.expand(len(codes),-1,-1))),None).batch["pixels"][:,1:]
            pred=np.stack([rgb(p) for p in predictions])
            metrics=[metric(p.astype(np.float32)/255,images[2:],[32])["metrics"]["32"] for p in pred]
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder / "candidates.npz", initial=images[:2], reference=images[2:],
                            prediction=pred, actions=actions, instruction=np.array(task))
        write_json(folder / "metadata.json",dict(id=row["id"],sha256=file_sha256(folder / "candidates.npz"),
                   implementation=implementation(),metrics=metrics,trace=trace,seconds=time.monotonic()-started))
        print(json.dumps(dict(event="calibration_candidates_ready",id=row["id"],seconds=time.monotonic()-started)),flush=True)
    del pipe, metric, worker
    gc.collect()
    torch.cuda.empty_cache()


def calibrate():
    cfg=prepare()
    runtime(api=True)
    store=ReferenceAlignmentStore({**cfg["decisions"],"max_api_calls":32})
    results=[]
    for row in load(EXP / "protocol.json")["calibration_rows"]:
        folder=EXP / "calibration" / row["id"]
        metadata=load(folder / "metadata.json")
        assert metadata["implementation"] == implementation()
        assert file_sha256(folder / "candidates.npz") == metadata["sha256"]
        with np.load(folder / "candidates.npz",allow_pickle=False) as z:
            initial, reference, predictions=z["initial"],z["reference"],z["prediction"]
            options={f"candidate_{i}":p for i,p in enumerate(predictions)}
            options.update(reference=reference, frozen=np.repeat(initial[-1:],32,axis=0), black=np.zeros_like(reference))
            scores={}
            for label, pred in options.items():
                key,score=store.score_arrays(initial,pred,z["actions"][2:],str(z["instruction"].item()),"robocasa",reference)
                scores[label]=dict(key=key,score=score)
                print(json.dumps(dict(event="calibration_score",id=row["id"],label=label,score=score,api_calls=store.client.calls)),flush=True)
                write_json(folder / "scores.json",scores)
        results.append(dict(id=row["id"],scores=scores,metrics=metadata["metrics"]))
    protocol=load(EXP / "protocol.json")
    gate=protocol["gate"]
    controls_pass=True
    informative=0
    directions=[]
    for result in results:
        s={k:v["score"] for k,v in result["scores"].items()}
        controls_pass &= (s["reference"]-s["frozen"]>=gate["reference_over_freeze_min"]
                          and s["reference"]-s["black"]>=gate["reference_over_black_min"])
        values=[s[f"candidate_{i}"] for i in range(4)]
        informative+=int(max(values)-min(values)>=gate["informative_candidate_range"])
        for i in range(4):
            for j in range(i+1,4):
                a,b=result["metrics"][i]["lpips"],result["metrics"][j]["lpips"]
                if abs(a-b)/max(a,b,1e-9)>=gate["decisive_metric_relative_gap"] and abs(values[i]-values[j])>=gate["decisive_reward_gap"]:
                    directions.append(dict(id=result["id"],pair=[i,j],correct=(values[i]-values[j])*(a-b)<0))
    correct=sum(d["correct"] for d in directions)
    passed=(controls_pass and informative>=gate["informative_groups_min"] and len(directions)>=gate["decisive_pairs_min"]
            and correct/max(1,len(directions))>=gate["minimum_lpips_direction_fraction"])
    report=dict(state="PASS" if passed else "CALIBRATION_FAILED",controls_pass=bool(controls_pass),
                informative_groups=informative,total_groups=len(results),decisive_pairs=len(directions),
                direction_correct=correct,api_calls_this_invocation=store.client.calls,
                results=results,directions=directions,gate=gate)
    write_json(EXP / "calibration.json",report)
    (EXP / "REPORT.md").write_text(
        "# Pure Decisions / local success SFT\n\n"
        f"Calibration: {report['state']}. Controls passed: {controls_pass}. Informative groups: {informative}/{len(results)}. "
        f"LPIPS direction agreement among decisive candidate pairs: {correct}/{len(directions)}.\n\n"
        "32 predicted frames from original episode frames 0/1. The judge sees 8 synchronized reference/prediction pairs. "
        "Reward is only the mean of three OpenAI Decisions scores; no MAE/LPIPS reward. GT is reviewer-only.\n\n"
        "Calibration uses four training initial-state groups, excluded from the four-update pilot schedule. "
        "The group/pair counts are small and correlated; passing this gate does not establish improvement. "
        "On failure, no RL is launched.\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("results","directions","gate")}),flush=True)
    return passed


def train_run(resume=False):
    from futureworlds.training import train
    cfg=prepare()
    if load(EXP / "calibration.json")["state"]!="PASS":
        raise RuntimeError("Calibration failed; no RL update authorized by this experimental protocol")
    runtime(api=True)
    path=EXP / "train.json"
    if resume:
        checkpoints=sorted((EXP / "train/checkpoints").glob("step_*"))
        cfg["resume"]=str(checkpoints[-1])
        path=EXP / "resume.json"
        write_json(path,cfg)
    train(path)


def evaluate_run():
    from futureworlds.evaluation import evaluate
    from futureworlds.exporting import export_model
    cfg=prepare()
    runtime()
    checkpoint=EXP / "train/checkpoints/step_000004"
    if not (EXP / "train/complete.json").exists():
        raise RuntimeError("Four-update pilot has not completed")
    bundle=EXP / "rl_bundle"
    export_model(cfg["bundle"],"robocasa",checkpoint,bundle)
    for label,source,variant in [("sft",cfg["bundle"],"sft"),("rl",bundle,"main")]:
        evaluate(str(source),"robocasa",variant,cfg["evaluation_manifest"],str(EXP / "evaluation" / label),
                 device="cuda",metric_device="cuda",text_model=cfg["text_model"],horizons=[10,20,32],beam_width=4)
        gc.collect()
        torch.cuda.empty_cache()
    sft=load(EXP / "evaluation/sft/summary.json")
    rl=load(EXP / "evaluation/rl/summary.json")
    comparison={h:{k:dict(sft=v,rl=rl["aggregate"][h][k],delta=rl["aggregate"][h][k]-v)
                    for k,v in values.items()} for h,values in sft["aggregate"].items()}
    write_json(EXP / "comparison.json",comparison)
    with (EXP / "REPORT.md").open("a") as f:
        f.write("\nFour-update pilot completed. Fixed four-group validation, no checkpoint selection.\n\n")
        f.write("Horizon | SFT LPIPS | RL LPIPS | Delta\n---|---|---|---\n")
        for h,c in comparison.items():
            m=c["lpips"];f.write(f"{h}|{m['sft']:.6f}|{m['rl']:.6f}|{m['delta']:+.6f}\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare","collect","calibrate","train","evaluate","all"])
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    EXP.mkdir(parents=True,exist_ok=True)
    try:
        if a.stage=="all":
            collect()
            if calibrate():
                train_run(a.resume)
                evaluate_run()
        elif a.stage=="prepare":prepare()
        elif a.stage=="collect":collect()
        elif a.stage=="calibrate":calibrate()
        elif a.stage=="train":train_run(a.resume)
        else:evaluate_run()
    except Exception as exc:
        # urllib authorization headers are never emitted.
        write_json(EXP / "status.json",dict(state="STOPPED",stage=a.stage,error_type=type(exc).__name__,error=str(exc)))
        raise

