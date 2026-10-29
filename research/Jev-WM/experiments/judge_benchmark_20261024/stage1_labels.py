"""Stage 1 (CPU): h5 event labels for every logged candidate + accuracy of logged judges.

Reference frame k of a horizon-32 RL window = packed frame 2+k = h5 index (2+k)*4.
GT lifted state per frame comes from h5 events/flags[:,lift]. Predicted lifted state
comes from the red-cube tracker (rise in px vs the last initial frame), with the px
threshold calibrated on GT reference frames against the h5 flag.
"""
import json, glob, itertools, sys
from pathlib import Path
import numpy as np, h5py

E = Path("/media/tyshao/Jev-WM/experiments")
OUT = E / "judge_benchmark_20261024"
RUNS = {"v1": E / "robocasa_object_memspo_4gpu50_20261024", "v2": E / "robocasa_object_memspo_4gpu_v2_20261024"}
MAN = {r["id"]: r for r in json.load(open("/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/train_natural.json"))["cases"]}
SCHED = json.load(open(RUNS["v1"] / "schedule.json"))
H = 32
IDX = (np.arange(H) + 2) * 4


def h5_state(row):
    with h5py.File(row["source_h5"]) as f:
        fl = f["events/flags"][:]
        z = f["observations/object_pose_wxyz"][:, 2]
    i = IDX[IDX < len(fl)]
    if len(i) < H:
        raise ValueError("short episode")
    return dict(lift=fl[i, 2].astype(int), grasp=fl[i, 1].astype(int), dz_cm=((z[i] - z[0]) * 100).round(2))


def ys(track):
    return np.array([np.nan if t["xy"] is None else t["xy"][1] for t in track])


def rise(track, y0):
    return y0 - ys(track)  # image y grows downward


groups = []
cache = {}
for run, root in RUNS.items():
    for f in sorted(glob.glob(str(root / "judge_log/[0-9]*[0-9].json"))):
        d = json.load(open(f))
        c = d["cursor"]
        if c >= len(SCHED):
            continue
        row = MAN[SCHED[c]["id"]]
        if row["id"] not in cache:
            cache[row["id"]] = h5_state(row)
        groups.append(dict(run=run, cursor=c, id=row["id"], success=row["success"], failure_type=row["failure_type"],
                           d=d, gt=cache[row["id"]]))

# Calibrate px threshold: GT tracker rise vs h5 lift flag (GT y0 = first reference frame, cube on table at t=2 in all cases)
r_all, l_all = [], []
for g in groups:
    t = g["d"]["tracks"][0]["gt"]
    y = ys(t)
    r_all += list(y[0] - y); l_all += list(g["gt"]["lift"])
r_all, l_all = np.array(r_all), np.array(l_all)
ok = ~np.isnan(r_all)
best = max(range(2, 40), key=lambda T: ((r_all[ok] > T) == l_all[ok]).mean())
acc = ((r_all[ok] > best) == l_all[ok]).mean()
print(f"calibrated rise threshold {best}px, GT-tracker vs h5 lift agreement {acc:.3f} on {ok.sum()} frames; "
      f"lifted frames {l_all.mean():.2f}")

rows = []
for g in groups:
    d, gt = g["d"], g["gt"]
    y0 = ys(d["tracks"][0]["gt"])[0]
    cands = []
    for k, t in enumerate(d["tracks"]):
        r = rise(t["pred"], y0)
        pl = np.where(np.isnan(r), 0, r > best).astype(int)
        missing = np.isnan(r) & ~np.isnan(ys(t["gt"]))
        agree = float((pl == gt["lift"]).mean())
        cands.append(dict(k=k, agree=agree, fake_lift=int(((pl == 1) & (gt["lift"] == 0)).sum()),
                          miss_lift=int(((pl == 0) & (gt["lift"] == 1)).sum()), missing=int(missing.sum()),
                          rule=d["raw_reward"][k] if "interaction" not in d["weights"] else
                          (0.5 * d["axes"]["position"][k] + 0.25 * d["axes"]["persistence"][k]) / 0.75,
                          position=d["axes"]["position"][k], persistence=d["axes"]["persistence"][k],
                          vlm=d["interaction_raw"][k], total=d["raw_reward"][k]))
    rows.append(dict(run=g["run"], cursor=g["cursor"], id=g["id"], success=g["success"], failure_type=g["failure_type"],
                     gt_lift_frames=int(gt["lift"].sum()), cands=cands))

(OUT / "labels.json").write_text(json.dumps(dict(threshold_px=best, calib_acc=acc, groups=rows), indent=1))

# Pairs: within group, event-agreement differs by >= 0.25 (>= 8 of 32 frames)
def pairs(sel):
    out = []
    for g in sel:
        for a, b in itertools.combinations(g["cands"], 2):
            if abs(a["agree"] - b["agree"]) >= 0.25:
                good, bad = (a, b) if a["agree"] > b["agree"] else (b, a)
                out.append((g, good, bad))
    return out

def report(name, sel):
    P = pairs(sel)
    contrast = np.mean([any(abs(a["agree"] - b["agree"]) >= 0.25 for a, b in itertools.combinations(g["cands"], 2)) for g in sel])
    line = f"{name:34s} groups {len(sel):4d} contrast {contrast:.2f} pairs {len(P):4d}"
    for key in ("rule", "position", "persistence", "vlm", "total"):
        if P:
            line += f" | {key} {np.mean([gd[key] > bd[key] for _, gd, bd in P]):.2f}"
    print(line)

for run in ("v1", "v2"):
    S = [g for g in rows if g["run"] == run]
    print(f"== {run} (vlm = that run's interaction prompt; rule = position/persistence only)")
    report("all", S)
    report("GT no lift in window", [g for g in S if g["gt_lift_frames"] == 0])
    report("GT lifts in window", [g for g in S if g["gt_lift_frames"] > 0])
    for ft in sorted({g["failure_type"] or "success" for g in S}):
        report("  " + ft, [g for g in S if (g["failure_type"] or "success") == ft])
