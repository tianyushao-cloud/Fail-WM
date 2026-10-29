"""Bounded reliable-pair pilot; pure Decisions, fixed protocol, server only."""
import argparse,json,gc
from pathlib import Path
import numpy as np
import torch
import focused_decisions as base
from futureworlds.reliable_pair import select_pair,POLICY,COMPONENTS
from futureworlds.focused_alignment import FocusedStore,QUESTIONS,PROMPT
from futureworlds.data import read_manifest,write_json
from futureworlds.protocol import implementation
from futureworlds.checkpoints import file_sha256
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"runs/local_success_decisions"
EXP=PRIOR/"reliable_pair"
load=base.load

def prepare():
 EXP.mkdir(exist_ok=True)
 if (EXP/"protocol.json").exists():return load(EXP/"train.json")
 cfg=load(ROOT/"runs/local_success_memspo_h32/train.json")
 rows=load(PRIOR/"framewise/protocol.json")["calibration_rows"]
 train=read_manifest(cfg["train_manifest"],"robocasa")["cases"];index={r["id"]:r for r in train}
 excluded={r["initial_state_id"] for r in rows}
 for p in (PRIOR/"protocol.json",PRIOR/"focused/protocol.json",PRIOR/"focused/initial_protocol.json"):
  excluded|={r["initial_state_id"] for r in load(p)["calibration_rows"]}
 schedule=[e for e in load(ROOT/"runs/local_success_memspo_h32/schedule.json") if index[e["id"]]["initial_state_id"] not in excluded][:8]
 cfg.update(reward="decisions",steps=4,global_batch=2,checkpoint_every=1,
   schedule=str(EXP/"schedule.json"),output=str(EXP/"train"),paper_locked=False,
   evaluation_manifest=str(PRIOR/"validation/manifest.json"),
   decisions=dict(backend="api",model="gpt-6-luna",rubric="reliable_pair",review_dir=str(EXP/"reviews"),
                  max_api_calls=72,timeout_seconds=90,retries=0,spacing_seconds=3))
 gate=dict(clean_min=.8,black_max=.1,freeze_gap=.1,object_corruption_gap=.15,
           informative_groups=2,maximum_selected_lpips_ratio=1.05)
 protocol=dict(baseline=cfg["bundle"],baseline_sha256=file_sha256(Path(cfg["bundle"])/"manifest.json"),
  initial_frames=[0,1],horizon=32,calibration_rows=rows,selection=POLICY,gate=gate,questions=QUESTIONS,prompt=PROMPT,
  selection_logic="one pair per group; object and composite gaps >=.04 in BOTH orders; no fidelity/scene tradeoff; each selected candidate component order delta<=.15; winner object>=.50",
  reward="winner +1; loser -1; others 0. Zero mean gives unselected candidates exactly zero policy-gradient advantage. Original KL still applies.",
  calibration_budget=56,train_budget=72,steps=4,global_batch=2,
  validation_manifest=cfg["evaluation_manifest"],checkpoint_selection="fixed final step4",
  limitations="Exploratory training-split calibration; unchanged failed focused scorer is reused with stricter component-wise pair eligibility. It may still fail; no claim that ordering consistency proves correctness.",
  implementation=implementation())
 write_json(EXP/"train.json",cfg);write_json(EXP/"schedule.json",schedule);write_json(EXP/"protocol.json",protocol)
 return cfg

def collect():
 base.EXP=EXP;base.prepare=prepare;base.collect()

def calibrate():
 cfg=prepare();base.runtime(api=True);protocol=load(EXP/"protocol.json");results=[]
 store=FocusedStore({**cfg["decisions"],"max_api_calls":56})
 for row in protocol["calibration_rows"]:
  folder=EXP/"calibration"/row["id"];meta=load(folder/"metadata.json")
  assert meta["implementation"]==implementation()
  assert meta["sha256"]==file_sha256(folder/"candidates.npz")
  scores=base.score_case(store,folder/"candidates.npz",folder)
  controls=base.control_check(scores,protocol["gate"])
  x=[[[scores[f"candidate_{i}"]["orders"][o]["components"][n] for n in COMPONENTS] for i in range(4)] for o in range(2)]
  rewards,selection=select_pair(x);chosen=selection["chosen"]
  lp=[m["lpips"] for m in meta["metrics"]]
  result=dict(id=row["id"],controls=controls,selection=selection,rewards=rewards.tolist(),metrics=meta["metrics"],
              selected_lpips=lp[chosen["winner"]] if chosen else None,mean_lpips=float(np.mean(lp)))
  results.append(result);write_json(folder/"pair_selection.json",result)
  print(json.dumps(dict(event="calibration_group",id=row["id"],chosen=chosen,controls_passed=controls["passed"])),flush=True)
 accepted=[r for r in results if not r["selection"]["rejected"]]
 ratio=sum(r["selected_lpips"] for r in accepted)/sum(r["mean_lpips"] for r in accepted) if accepted else None
 passed=all(r["controls"]["passed"] for r in results) and len(accepted)>=protocol["gate"]["informative_groups"] and ratio<=1.05
 report=dict(state="PASS" if passed else "CALIBRATION_FAILED",results=results,informative_groups=len(accepted),
             selected_to_mean_lpips_ratio=ratio,new_api_calls=store.client.calls,implementation=implementation())
 write_json(EXP/"calibration.json",report)
 print(json.dumps({k:v for k,v in report.items() if k not in ("results","implementation")}),flush=True)

def train():
 cfg=prepare();base.runtime(api=True)
 audit=load(EXP/"calibration.json")
 if audit["state"]!="PASS" or audit["implementation"]!=implementation():raise RuntimeError("No passing calibration")
 if not (EXP/"visual_audit.json").exists():raise RuntimeError("Selected-pair visual audit required")
 visual=load(EXP/"visual_audit.json")
 if not visual.get("passed") or visual.get("calibration_sha256")!=file_sha256(EXP/"calibration.json") or visual.get("protocol_sha256")!=file_sha256(EXP/"protocol.json"):
  raise RuntimeError("Visual audit missing or stale")
 from futureworlds.training import train as run
 run(EXP/"train.json")

def evaluate():
 base.EXP=EXP;base.prepare=prepare;base.evaluate_run()

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","collect","calibrate","audit_run","train","evaluate"]);args=p.parse_args()
 prepare()
 if args.stage=="audit_run":collect();calibrate()
 else:globals()[args.stage]()

