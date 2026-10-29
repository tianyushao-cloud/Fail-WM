"""Stage B report (CPU): within-group ranking of RL rollouts, logged v0 state axis vs new variants.

Truth per candidate = agreement of its tracker lift timeline with the h5 lift timeline (32 frames).
Pairs: same group, agreement differs >= 0.25 (stage-1 benchmark definition). Also reported: the
binary fake-lift contrast in GT-no-lift groups (tracker no-rise candidate should beat a rise candidate).
reward = 0.35 position + 0.15 persistence + 0.5 state (training weights; position/persistence logged).
"""
import collections, itertools, json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
RUN = Path("/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/judge_log")
THR = 8
TS = sorted(set(range(0, 32, 2)) | {31})
groups = json.loads((D / "stage_b_groups.json").read_text())
res = collections.defaultdict(dict)
for l in (D / "results" / "stage_b.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l); res[(r["variant"], r["cursor"])][r["k"]] = np.array(r["p_lift"])
variants = ["v0"] + sorted({v for v, _ in res})

rows = []
for c in groups:
    d = json.loads((RUN / f"{c:06d}.json").read_text())
    gt = np.array(d["gt_lift"])
    tl = []
    for t in d["tracks"]:
        y0 = next(g["xy"][1] for g in t["gt"] if g["xy"] is not None)
        ys = np.array([np.nan if p["xy"] is None else p["xy"][1] for p in t["pred"]])
        tl.append(np.where(np.isnan(ys), 0, (y0 - ys) > THR).astype(int))
    tl = np.array(tl)
    rule = 0.35 * np.array(d["axes"]["position"]) + 0.15 * np.array(d["axes"]["persistence"])
    st = {"v0": np.array(d["axes"]["state"])}
    plift = {"v0": np.array(d["p_lift"])}
    for v in variants[1:]:
        if len(res.get((v, c), {})) == len(tl):
            plift[v] = np.array([res[(v, c)][k] for k in range(len(tl))])
            st[v] = 1 - np.abs(plift[v] - gt[TS][None]).mean(1)
    rows.append(dict(c=c, gt_any=bool(gt.any()), agree=(tl == gt[None]).mean(1), rise=tl.sum(1) >= 3,
                     rule=rule, st=st, plift=plift, tl=tl[:, TS]))
have = [v for v in variants if all(v in r["st"] for r in rows)]
print(f"{len(rows)} groups ({sum(not r['gt_any'] for r in rows)} GT-no-lift mixed, {sum(r['gt_any'] for r in rows)} GT-lift); "
      f"variants complete on all groups: {have}")


def acc(sel, score):
    out = []
    for r in sel:
        s = score(r)
        for i, j in itertools.combinations(range(len(r["agree"])), 2):
            if abs(r["agree"][i] - r["agree"][j]) >= 0.25:
                g, b = (i, j) if r["agree"][i] > r["agree"][j] else (j, i)
                out.append(float(s[g] > s[b]) + 0.5 * float(s[g] == s[b]))
    return np.mean(out) if out else float("nan"), len(out)


def fake(sel, score):
    out = []
    for r in sel:
        s = score(r)
        for i in np.where(~r["rise"])[0]:
            for j in np.where(r["rise"])[0]:
                out.append(float(s[i] > s[j]) + 0.5 * float(s[i] == s[j]))
    return np.mean(out) if out else float("nan"), len(out)


segs = [("GT no lift", [r for r in rows if not r["gt_any"]]), ("GT lifts", [r for r in rows if r["gt_any"]]), ("ALL", rows)]
print("\nwithin-group pair accuracy (h5-better candidate scores higher)")
print(f"{'':12s} {'rule only':>14s}" + "".join(f"{'state ' + v:>14s}{'reward ' + v:>14s}" for v in have))
for name, sel in segs:
    a, n = acc(sel, lambda r: r["rule"])
    line = f"{name:12s} {a:10.3f} n={n:<3d}"
    for v in have:
        line += f"{acc(sel, lambda r: r['st'][v])[0]:14.3f}{acc(sel, lambda r: r['rule'] + 0.5 * r['st'][v])[0]:14.3f}"
    print(line)
no = segs[0][1]
print("\nGT-no-lift fake-lift contrast: tracker no-rise candidate beats rise candidate")
print(f"   rule only {fake(no, lambda r: r['rule'])[0]:.3f} (n={fake(no, lambda r: r['rule'])[1]})" +
      "".join(f" | state {v} {fake(no, lambda r: r['st'][v])[0]:.3f}, reward {v} {fake(no, lambda r: r['rule'] + 0.5 * r['st'][v])[0]:.3f}" for v in have))
print("\ncalibration on rollout frames vs tracker lift (sampled timestamps), GT-no-lift groups")
for v in have:
    P = np.concatenate([r["plift"][v].ravel() for r in no]); T = np.concatenate([r["tl"].ravel() for r in no])
    print(f"   {v}: frames tracker-flat called lifted {(P[T == 0] > .5).mean():.2f}, tracker-rise called lifted {(P[T == 1] > .5).mean():.2f}")
print("\nstate-axis spread within group (median ptp x 0.5 weight), and groups whose total reward spread < 0.02")
for v in have:
    for name, sel in segs[:2]:
        sp = [0.5 * np.ptp(r["st"][v]) for r in sel]
        flat = np.mean([np.ptp(r["rule"] + 0.5 * r["st"][v]) < 0.02 for r in sel])
        print(f"   {v} {name:10s} state spread {np.median(sp):.3f}  flat groups {flat:.2f}")
