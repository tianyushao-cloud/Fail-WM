import pickle, itertools, collections, numpy as np
from traj import aggregate, CAP
G = pickle.load(open("groups.pkl", "rb"))
def T(w):
    return lambda c: 1 - aggregate(w[0] * c["err"]["cube"] + w[1] * c["err"]["grip"] + w[2] * c["err"]["rel"]) / CAP
S = {"rule(.35pos+.15pers)": lambda c: .35 * c["position"] + .15 * c["persistence"],
     "state_v0(VLM)": lambda c: c["state"], "v0_train_total": lambda c: c["v0"],
     "T_cube": T((1, 0, 0)), "T_grip": T((0, 1, 0)), "T_rel": T((0, 0, 1)),
     "T_cube+grip": T((.67, .33, 0)), "T_full(.5/.25/.25)": T((.5, .25, .25))}
def pairs(sel, label, margin):
    out = collections.defaultdict(list)
    for g in sel:
        lab = label(g); sc = {k: np.array([f(c) for c in g["cands"]]) for k, f in S.items()}
        for i, j in itertools.combinations(range(len(lab)), 2):
            if abs(lab[i] - lab[j]) >= margin:
                a, b = (i, j) if lab[i] > lab[j] else (j, i)
                for k, s in sc.items(): out[k].append(float(s[a] > s[b]) + .5 * float(s[a] == s[b]))
    return out
agree = lambda g: np.array([(c["tl"] == g["gt_lift"]).mean() for c in g["cands"]])
art = lambda g: -np.array([c["dup"] + c["vanish"] + c["jumps"] for c in g["cands"]], float)
def show(title, sel, label, margin):
    o = pairs(sel, label, margin); n = len(next(iter(o.values()), []))
    print(f"{title:34s} n={n:5d} | " + " ".join(f"{np.mean(v):.2f}" for v in o.values()))
print(" " * 43 + "| " + " ".join(k for k in S))
nolift = [g for g in G if not g["gt_lift"].any()]; lift = [g for g in G if g["gt_lift"].any()]
print("== A. lift-timeline label (tracker vs h5, diff>=.25; benchmark def, partly circular for cube terms)")
show("ALL", G, agree, .25); show("GT no lift", nolift, agree, .25); show("GT lifts", lift, agree, .25)
for ft in sorted({g["failure_type"] or "success" for g in G}):
    show("  " + ft, [g for g in G if (g["failure_type"] or "success") == ft], agree, .25)
print("== B. artifact label (fewer dup+vanish+jump frames, diff>=4), restricted to pairs with equal lift agreement (+-.05)")
def art_eq(sel):
    out = collections.defaultdict(list)
    for g in sel:
        a, ag = art(g), agree(g); sc = {k: np.array([f(c) for c in g["cands"]]) for k, f in S.items()}
        for i, j in itertools.combinations(range(len(a)), 2):
            if abs(a[i] - a[j]) >= 4 and abs(ag[i] - ag[j]) <= .05:
                x, y = (i, j) if a[i] > a[j] else (j, i)
                for k, s in sc.items(): out[k].append(float(s[x] > s[y]) + .5 * float(s[x] == s[y]))
    TITLE = "equal-lift artifact pairs"; n = len(next(iter(out.values()), [])); print(f"{TITLE:34s} n={n:5d} | " + " ".join(f"{np.mean(v):.2f}" for v in out.values()))
art_eq(G)
print("== C. group contrast: fraction of groups with score range > .02 / median range")
for k, f in S.items():
    r = np.array([np.ptp([f(c) for c in g["cands"]]) for g in G]); rn = np.array([np.ptp([f(c) for c in g["cands"]]) for g in nolift])
    print(f"  {k:22s} all {np.mean(r>.02):.2f} med {np.median(r):.3f} | GT-no-lift {np.mean(rn>.02):.2f} med {np.median(rn):.3f}")
print("== D. GT-no-lift groups where EVERY candidate fake-lifts (>=3 risen frames): rank by fewer fake-lift frames")
allfake = [g for g in nolift if all(c["tl"].sum() >= 3 for c in g["cands"])]
print(f"  {len(allfake)} / {len(nolift)} GT-no-lift groups are all-fake")
show("all-fake, fewer fake frames (>=3)", allfake, lambda g: -np.array([c["tl"].sum() for c in g["cands"]], float), 3)
print("== E. mean per-frame errors (px)")
for nm, sel in (("GT no lift", nolift), ("GT lifts", lift)):
    e = {k: np.mean([c["err"][k].mean() for g in sel for c in g["cands"]]) for k in ("cube", "grip", "rel", "match")}
    print(f"  {nm}: " + " ".join(f"{k} {v:.2f}" for k, v in e.items()))
