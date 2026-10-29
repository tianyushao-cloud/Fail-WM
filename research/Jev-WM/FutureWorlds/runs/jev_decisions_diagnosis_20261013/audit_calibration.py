"""Recheck consolidated calibration evidence using original RGB; no API calls.
Run from repository: PYTHONPATH=src .venv/bin/python runs/jev_decisions_diagnosis_20261013/audit_calibration.py
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from futureworlds.evaluation import Metrics

torch.set_num_threads(4)
root=Path(__file__).resolve().parents[2];evidence=Path(__file__).with_name("CALIBRATION_EVIDENCE.json");e=json.loads(evidence.read_text());m=Path(e["manifest"]);refs={}
for r in json.loads(m.read_text())["cases"]:
 with np.load(m.parent/r["file"],allow_pickle=False) as z:
  k=hashlib.sha256(np.ascontiguousarray(z["images"][:2]).tobytes()).hexdigest()
  if k in refs:assert np.array_equal(refs[k],z["images"][2:6]),"ambiguous future"
  refs[k]=z["images"][2:6].copy()
metric=Metrics("cpu");cache={}
for name,exp in e["experiments"].items():
 for pair in exp["pairs"]:
  for side,row in pair["sides"].items():
   key=row["source"]
   if key not in cache:
    file=root/"runs/jev_pure_20261012/reviews"/key/"frames.npz"
    with np.load(file,allow_pickle=False) as z:
     assert hashlib.sha256(np.ascontiguousarray(z["initial_rgb"]).tobytes()).hexdigest()==pair["state"]
     pred=z["prediction_rgb"].astype(np.float32)/255
    cache[key]=metric(pred,refs[pair["state"]],[4])["metrics"]["4"]
   for k in ["mae","lpips"]:assert abs(cache[key][k]-row["metrics"][k])<1e-6,(name,key,k)
 for expected in exp["thresholds"]:
  selected=[r for r in exp["pairs"] if abs(r["reward_delta"])>=expected["threshold"]];assert len(selected)==expected["pairs"]
  for k in ["mae","lpips"]:
   products=[r["reward_delta"]*(cache[r["sides"]["high"]["source"]][k]-cache[r["sides"]["low"]["source"]][k]) for r in selected]
   result=dict(correct=sum(v<0 for v in products),wrong=sum(v>0 for v in products),ties=sum(v==0 for v in products));assert result==expected["direction"][k]
 print(name,json.dumps(exp["thresholds"][0]),flush=True)
print("PASS: cached metrics and reward directions reproduced; no API calls",flush=True)
