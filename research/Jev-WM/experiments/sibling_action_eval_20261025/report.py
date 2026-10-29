"""Sibling-action report. A frame counts as lifted when the red cube is >=8px above its
initial position (threshold calibrated vs the h5 lift flag: 98.9% agreement on GT frames)."""
import json, sys, itertools, collections
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
THR = 8


def load(label):
    out = []
    for f in sorted((R / "results" / label).glob("g*_b*.json")):
        d = json.loads(f.read_text())
        r = np.array([np.nan if v is None else v for v in d["pred_rise_px"]], dtype=float)
        d["pred_lift"] = np.where(np.isnan(r), 0, r > THR).astype(int)
        d["gt_lift"] = np.array(d["gt_lift"])
        d["gt_any"] = int(d["gt_lift"].any())
        d["pred_any"] = int(d["pred_lift"].sum() >= 3)
        d["frame_agree"] = float((d["pred_lift"] == d["gt_lift"]).mean())
        d["missing"] = float(np.isnan(r).mean())
        out.append(d)
    return out


def summarize(label, rows):
    gl = np.array([d["gt_any"] for d in rows]); pl = np.array([d["pred_any"] for d in rows])
    by = collections.defaultdict(list)
    for d in rows:
        by[d["group"]].append(d)
    disc = [];same_ok = [];lift_only = []
    for g in by.values():
        for a, b in itertools.combinations(g, 2):
            if a["gt_any"] != b["gt_any"]:
                lift, nol = (a, b) if a["gt_any"] else (b, a)
                disc.append(lift["pred_any"] > nol["pred_any"])
    groups_diff = sum(len({d["pred_any"] for d in g}) > 1 for g in by.values())
    print(f"== {label}: {len(rows)} trajectories, {len(by)} initial states")
    print(f"  window-level lift prediction accuracy      {np.mean(gl == pl):.3f}")
    print(f"  GT lifts      -> predicted lift (recall)    {pl[gl == 1].mean():.3f}  (n={int((gl == 1).sum())})")
    print(f"  GT no lift    -> predicted lift (OPTIMISM)  {pl[gl == 0].mean():.3f}  (n={int((gl == 0).sum())})")
    print(f"  sibling pairs with different GT outcome: predicted outcome differs in the right direction "
          f"{np.mean(disc):.3f} (n={len(disc)}; 0.5 = ignores action)")
    print(f"  initial states where predictions differ across siblings: {groups_diff}/{len(by)}")
    print(f"  per-frame lift agreement {np.mean([d['frame_agree'] for d in rows]):.3f}; "
          f"cube missing in pred {np.mean([d['missing'] for d in rows]):.3f}")
    ft = collections.defaultdict(list)
    for d in rows:
        ft[(d["failure_type"] or "success", d["gt_any"])].append(d["pred_any"])
    print("  predicted-lift rate by type (gt_lift_in_window):",
          {f"{k[0]}({k[1]})": f"{np.mean(v):.2f}/{len(v)}" for k, v in sorted(ft.items())})
    return {d["id"]: d for d in rows}


labels = sys.argv[1:] or sorted(p.name for p in (R / "results").iterdir() if p.is_dir())
res = {l: summarize(l, load(l)) for l in labels}
gt = next(iter(res.values()))
gl = np.array([d["gt_any"] for d in gt.values()])
print(f"\nreference baselines on the same {len(gl)} windows: always-lift accuracy {gl.mean():.3f}; "
      f"'full gripper close => lift' accuracy {gl.mean():.3f} (every window closes fully)")
if len(labels) == 2:
    a, b = labels
    common = sorted(set(res[a]) & set(res[b]))
    fixed = sum(res[a][k]["pred_any"] != res[a][k]["gt_any"] and res[b][k]["pred_any"] == res[b][k]["gt_any"] for k in common)
    broke = sum(res[a][k]["pred_any"] == res[a][k]["gt_any"] and res[b][k]["pred_any"] != res[b][k]["gt_any"] for k in common)
    print(f"\n{a} -> {b} on {len(common)} common windows: fixed {fixed}, broke {broke}")
