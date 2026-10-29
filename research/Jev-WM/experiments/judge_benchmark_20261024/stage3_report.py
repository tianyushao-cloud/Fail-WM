"""Stage 3 (CPU): within-group ranking accuracy per judge on the stage-1 contrast pairs."""
import json, sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
lab = json.load(open(OUT / "labels.json"))
P = json.load(open(OUT / "pairs.json"))
lift = {k: np.array(v) for k, v in json.load(open(OUT / "lift_gt.json")).items()}
G = {(g["run"], g["cursor"]): g for g in lab["groups"]}
TS = sorted(set(range(0, 32, 2)) | {31})
R = lambda n: {json.loads(l)["key"]: np.array(json.loads(l)["p"]) for l in (OUT / "results" / f"{n}.jsonl").read_text().splitlines() if l.strip()} if (OUT / "results" / f"{n}.jsonl").exists() else {}
state, pair, point = R("state"), R("pair"), R("point_recon")

# State QA accuracy on GT frames vs h5 lift flag (cube missing counted as 'not lifted')
for src in ("raw", "recon"):
    y, p = [], []
    for k, v in state.items():
        f = k.split("|")
        if f[0] == "gt" and f[1] == src:
            y.append(lift[f[2]][int(f[3])]); p.append(v[1])
    if y:
        y, p = np.array(y), np.array(p)
        print(f"state-QA on GT {src:5s} frames: n={len(y)} acc={((p > .5) == y).mean():.3f} "
              f"lifted-recall={(p[y == 1] > .5).mean():.3f} table-recall={(p[y == 0] <= .5).mean():.3f}")

def state_score(run, cursor, k, cid):
    ps = [state.get(f"pred|{run}|{cursor}|{k}|{t}") for t in TS]
    if any(x is None for x in ps):
        return None
    pl = np.array([x[1] for x in ps]); gl = lift[cid][TS]
    return float(1 - np.abs(pl - gl).mean())  # soft agreement with h5 lift timeline

def point_score(run, cursor, k):
    v = [point.get(f"{run}|{cursor}|{k}|{o}") for o in ("reference_first", "prediction_first")]
    return None if any(x is None for x in v) else float(np.mean([(x * np.arange(5)).sum() / 4 for x in v]))

rows = []
for p in P:
    g = G[(p["run"], p["cursor"])]
    c = {x["k"]: x for x in g["cands"]}
    good, bad = c[p["good"]], c[p["bad"]]
    r = dict(seg="GT no lift" if p["gt_lift_frames"] == 0 else "GT lifts", ft=p["failure_type"], run=p["run"])
    r["rule"] = good["rule"] > bad["rule"]
    r["vlm_logged"] = good["vlm"] > bad["vlm"]
    sg, sb = state_score(p["run"], p["cursor"], p["good"], p["id"]), state_score(p["run"], p["cursor"], p["bad"], p["id"])
    r["state"] = None if sg is None else sg > sb
    pg, pb = point_score(p["run"], p["cursor"], p["good"]), point_score(p["run"], p["cursor"], p["bad"])
    r["point_recon"] = None if pg is None else pg > pb
    a = pair.get(f"{p['run']}|{p['cursor']}|{p['good']}|{p['bad']}|good_is_A")
    b = pair.get(f"{p['run']}|{p['cursor']}|{p['good']}|{p['bad']}|good_is_B")
    if a is not None and b is not None:
        win = (a[0] + b[1]) / 2  # P(good preferred), order-averaged
        r["pair"] = win > .5
        r["pair_order_flip"] = (a[0] > .5) != (b[1] > .5)
        r["pair_bias_A"] = (a[0] + b[0]) / 2
    else:
        r["pair"] = None
    if sg is not None:
        r["rule+state"] = (0.6 * good["position"] + 0.4 * good["persistence"]) * 0.7 + 0.3 * sg > \
                          (0.6 * bad["position"] + 0.4 * bad["persistence"]) * 0.7 + 0.3 * sb
    rows.append(r)

keys = ["rule", "vlm_logged", "point_recon", "pair", "state", "rule+state"]
def show(name, S):
    line = f"{name:28s} n={len(S):4d}"
    for k in keys:
        v = [r[k] for r in S if r.get(k) is not None]
        line += f" | {k} {np.mean(v):.2f}" if v else f" | {k}  -  "
    print(line)
print("\nwithin-group ranking accuracy (fraction of contrast pairs where the h5-better candidate scores higher)")
show("ALL", rows)
for seg in ("GT no lift", "GT lifts"):
    show(seg, [r for r in rows if r["seg"] == seg])
for ft in sorted({r["ft"] for r in rows}):
    show("  " + ft, [r for r in rows if r["ft"] == ft])
pr = [r for r in rows if r.get("pair") is not None]
if pr:
    print(f"\npair judge: order-inconsistent {np.mean([r['pair_order_flip'] for r in pr]):.2f}, mean P(answer A) {np.mean([r['pair_bias_A'] for r in pr]):.2f}")
