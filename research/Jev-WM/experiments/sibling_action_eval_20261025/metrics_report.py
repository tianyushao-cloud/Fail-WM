"""Paired frame-metric comparison vs a base label; bootstrap CI over initial states (the independent unit).
Usage: metrics_report.py BASE LABEL [LABEL ...]"""
import json, sys
from pathlib import Path
import numpy as np

R = Path("/media/tyshao/Jev-WM/experiments/sibling_action_eval_20261025/results")
base, labels = sys.argv[1], sys.argv[2:]
load = lambda l: json.loads((R / l / "frame_metrics.json").read_text())
meta = {}
for f in (R / base).glob("g*_b*.json"):
    d = json.loads(f.read_text())
    meta[d["id"]] = dict(group=d["group"], gt=int(any(d["gt_lift"])))
B = load(base); ids = sorted(B)
groups = sorted({meta[i]["group"] for i in ids})
rng = np.random.default_rng(0)
boot = [rng.choice(len(groups), len(groups)) for _ in range(5000)]


def agg(M, key, h, sel):
    return {i: float(np.mean(M[i][key][:h])) for i in sel}


print(f"{len(ids)} cases / {len(groups)} initial states; bootstrap over initial states, 95% CI of (label - {base})")
for seg, sel in (("all", ids), ("GT lift", [i for i in ids if meta[i]["gt"]]), ("GT no lift", [i for i in ids if not meta[i]["gt"]])):
    print(f"\n== {seg} (n={len(sel)})")
    for key, better in (("lpips", "lower"), ("psnr", "higher"), ("ssim", "higher")):
        for h in (8, 32):
            b = agg(B, key, h, sel)
            line = f"  {key:5s} {h:2d}f  {base} {np.mean(list(b.values())):.4f}"
            for l in labels:
                L = agg(load(l), key, h, sel)
                d = {g: [L[i] - b[i] for i in sel if meta[i]["group"] == g] for g in groups}
                d = {g: v for g, v in d.items() if v}
                G = list(d)
                stat = lambda idx: np.mean(np.concatenate([d[G[k]] for k in idx if k < len(G)]))
                cis = np.percentile([stat(rng.choice(len(G), len(G))) for _ in range(2000)], [2.5, 97.5])
                diff = np.mean([L[i] - b[i] for i in sel])
                better_n = sum((L[i] < b[i]) if better == "lower" else (L[i] > b[i]) for i in sel)
                line += f" | {l} {np.mean(list(L.values())):.4f} ({diff:+.4f} [{cis[0]:+.4f},{cis[1]:+.4f}], better {better_n}/{len(sel)})"
            print(line)
