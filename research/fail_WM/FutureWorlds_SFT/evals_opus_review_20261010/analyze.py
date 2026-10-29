#!/usr/bin/env python3
"""Event-anchored re-analysis of the 720-output E1 run (CPU only, read-only inputs)."""
import json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

ROOT = Path("/media/tyshao/fail_WM/FutureWorlds_SFT")
E1 = ROOT / "evals_e1_expanded_20261009"
OUT = ROOT / "evals_opus_review_20261010"
SRC = Path("/media/tyshao/fail_WM/RoboCasa/data/batch1000/manifest.jsonl")
DIV_PX, RUN = 8.0, 2

def center(f):
    f = f.astype(np.float32); r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y, x = np.where((r > 75) & (r > 1.5 * g) & (r > 1.4 * b))
    return (float(x.mean()), float(y.mean())) if len(x) >= 8 else None

sel = json.loads((E1 / "selection.json").read_text())
cases = {c["id"]: c for c in sel["cases"]}
src = {}
for line in SRC.read_text().splitlines():
    r = json.loads(line)
    if r["episode_id"] in cases: src[r["episode_id"]] = r

gt = {}
for cid, c in cases.items():
    imgs = np.load(ROOT / "robocasa_data" / c["file"])["images"]
    gt[cid] = [center(f) for f in imgs]

def phase_frames(r):
    out = {}
    for name, rng in r["phase_ranges"].items():
        if not name.startswith("attempt0/"): continue
        out[name.split("/", 1)[1]] = (rng["first_action"] / 4, rng["last_action"] / 4)
    return out

def phase_at(ph, t):
    for name, (a, b) in ph.items():
        if a <= t <= b + 0.75: return name
    return "after_end" if t > max(b for _, b in ph.values()) else "unknown"

def onset(pred, truth, start):
    run_d = run_m = 0; div = van = None
    for t in range(start, len(truth)):
        p, g = pred[t - 2], truth[t]
        d = p is not None and g is not None and math.hypot(p[0]-g[0], p[1]-g[1]) > DIV_PX
        m = p is None and g is not None
        run_d = run_d + 1 if d else 0; run_m = run_m + 1 if m else 0
        if div is None and run_d >= RUN: div = t - RUN + 1
        if van is None and run_m >= RUN: van = t - RUN + 1
    return div, van

def rise(track, start_xy):
    ys = [p[1] for p in track if p is not None]
    return (start_xy[1] - min(ys)) if ys else None

# GT calibration of the rise proxy: lifted vs non-lifted episodes
gt_rise = {cid: rise(gt[cid][2:], gt[cid][0]) for cid in cases}
lifted = [gt_rise[c] for c in cases if src[c]["event_summary"]["lift"]]
unlifted = [gt_rise[c] for c in cases if not src[c]["event_summary"]["lift"]]
thr = (max(unlifted) + min(lifted)) / 2 if unlifted and lifted and max(unlifted) < min(lifted) else None

recs = [json.loads(l) for l in (E1 / "per_prediction_metrics.jsonl").read_text().splitlines()]
rows = []
for r in recs:
    cid = r["id"]; raw = r["raw"]; track = [p["xy"] for p in raw["tracking"]]
    assert len(track) == len(gt[cid]) - 2, cid
    ph = phase_frames(src[cid])
    div, van = onset(track, gt[cid], 2)
    first = min([t for t in (div, van) if t is not None], default=None)
    rows.append(dict(model=r["model"], id=cid, group=r["initial_state_id"], success=r["success"],
        ftype=r["failure_type"], name=r["name"], decode="beam4" if r["name"] == "beam4" else "sampling",
        gt_lift=src[cid]["event_summary"]["lift"], gt_drop=src[cid]["event_summary"].get("drop"),
        cls=raw["class"], goalward=raw["goalward_by_margin"].get("2"), eligible=raw["goalward_eligible"],
        div=div, van=van, first=first, first_kind=None if first is None else ("diverge" if first == div else "vanish"),
        first_phase=None if first is None else phase_at(ph, first),
        rise=rise(track, gt[cid][0]), frames=len(gt[cid])))

def frac(xs): xs = [x for x in xs if x is not None]; return (sum(xs), len(xs))
report = {"rise_threshold_px": thr, "gt_rise_lifted": sorted(lifted), "gt_rise_unlifted": sorted(unlifted)}

# 1) case-level paired goalward with group bootstrap (sampling only)
rng = np.random.default_rng(0)
per = defaultdict(dict)
for x in rows:
    if x["decode"] == "sampling" and x["eligible"]:
        per[(x["group"], x["id"])].setdefault(x["model"], []).append(x["goalward"])
pairs = []
for (g, cid), d in per.items():
    s = [v for v in d["success"] if v is not None]; n = [v for v in d["natural"] if v is not None]
    pairs.append(dict(group=g, id=cid, success_all=sum(bool(v) for v in d["success"]) / 8,
                      natural_all=sum(bool(v) for v in d["natural"]) / 8,
                      success_vis=(sum(s) / len(s)) if s else None, natural_vis=(sum(n) / len(n)) if n else None))
groups = sorted({p["group"] for p in pairs})
def boot(key_a, key_b):
    diffs = []
    for _ in range(5000):
        gs = rng.choice(groups, len(groups))
        vals = [p[key_b] - p[key_a] for g in gs for p in pairs if p["group"] == g
                and p[key_a] is not None and p[key_b] is not None]
        if vals: diffs.append(np.mean(vals))
    pt = np.mean([p[key_b] - p[key_a] for p in pairs if p[key_a] is not None and p[key_b] is not None])
    return dict(point=float(pt), ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
report["paired_goalward_natural_minus_success"] = {"cases": len(pairs), "groups": len(groups),
    "all_outputs": boot("success_all", "natural_all"), "visible_only": boot("success_vis", "natural_vis"),
    "per_case": pairs}

# 2) where the first error starts (phase of first divergence/vanish), by model x GT outcome
tab = defaultdict(Counter)
for x in rows:
    key = f'{x["model"]}|{"success" if x["success"] else "failure"}|{x["decode"]}'
    tab[key][f'{x["first_kind"]}@{x["first_phase"]}' if x["first"] is not None else "never"] += 1
report["first_error_phase"] = {k: dict(v.most_common()) for k, v in sorted(tab.items())}

# 3) hallucinated lift on GT non-lift failures; carried-past-drop not attempted (needs 3D)
hl = defaultdict(lambda: [0, 0, 0])
for x in rows:
    if x["success"] or x["gt_lift"] or thr is None: continue
    k = f'{x["model"]}|{x["decode"]}|{x["ftype"]}'
    if x["rise"] is None: hl[k][2] += 1
    else: hl[k][0] += x["rise"] > thr; hl[k][1] += 1
report["hallucinated_lift_on_unlifted_failures"] = {k: dict(lift=v[0], visible=v[1], undetected=v[2]) for k, v in sorted(hl.items())}

# 4) vanish onset phase, by model x outcome
vt = defaultdict(Counter)
for x in rows:
    if x["van"] is not None:
        vt[f'{x["model"]}|{"success" if x["success"] else "failure"}'][phase_at(phase_frames(src[x["id"]]), x["van"])] += 1
report["vanish_onset_phase"] = {k: dict(v.most_common()) for k, v in sorted(vt.items())}

(OUT / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "paired_goalward_natural_minus_success"}, indent=1))
p = report["paired_goalward_natural_minus_success"]
print("paired:", p["cases"], "cases", p["groups"], "groups", "all", p["all_outputs"], "visible", p["visible_only"])
