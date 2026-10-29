"""Ranking report for trajectory-reward variants vs logged rewards (v0 8-GPU run, 480 groups x 8)."""
import pickle, itertools, collections, sys, numpy as np
from traj import score
G = pickle.load(open("groups.pkl", "rb"))
S = {"rule": lambda c: .35 * c["position"] + .15 * c["persistence"],
     "VLMst": lambda c: c["state"], "v0tot": lambda c: c["v0"]}
for cap in (40, 80):
    for van in (8, 15):
        S[f"Tc{cap}v{van}"] = (lambda cap, van: lambda c: score(c["err"], cap, van, (1, 0, 0)))(cap, van)
S["Tc+g"] = lambda c: score(c["err"], 80, 8, (.75, .25, 0))
S["Tc+r"] = lambda c: score(c["err"], 80, 8, (.5, 0, .5))
S["Tfull"] = lambda c: score(c["err"], 80, 8, (.5, .25, .25))
S["Tc+VLM"] = lambda c: .5 * score(c["err"], 80, 8, (1, 0, 0)) + .5 * c["state"]
for g in G:
    g["sc"] = {k: np.array([f(c) for c in g["cands"]]) for k, f in S.items()}
agree = lambda g: np.array([(c["tl"] == g["gt_lift"]).mean() for c in g["cands"]])
art = lambda g: -np.array([c["dup"] + c["vanish"] + c["jumps"] for c in g["cands"]], float)
def acc(sel, label, margin, cond=None):
    out = collections.defaultdict(list)
    for g in sel:
        lab = label(g)
        for i, j in itertools.combinations(range(len(lab)), 2):
            if abs(lab[i] - lab[j]) >= margin and (cond is None or cond(g, i, j)):
                a, b = (i, j) if lab[i] > lab[j] else (j, i)
                for k, s in g["sc"].items(): out[k].append(float(s[a] > s[b]) + .5 * float(s[a] == s[b]))
    return out
def show(title, o):
    n = len(next(iter(o.values()), []))
    print(f"{title:28s} n={n:4d} |" + "".join(f"{np.mean(v):7.2f}" for v in o.values()))
print(" " * 36 + "|" + "".join(f"{k:>7s}" for k in S))
nolift = [g for g in G if not g["gt_lift"].any()]; lift = [g for g in G if g["gt_lift"].any()]
print("A. lift-timeline label (diff>=.25)")
show("ALL", acc(G, agree, .25)); show("GT no lift", acc(nolift, agree, .25)); show("GT lifts", acc(lift, agree, .25))
for ft in sorted({g["failure_type"] or "success" for g in G}):
    sel = [g for g in G if (g["failure_type"] or "success") == ft]
    if acc(sel, agree, .25)["rule"]: show("  " + ft, acc(sel, agree, .25))
print("B. artifacts (dup+vanish+jump diff>=4, lift agreement equal +-.05)")
show("artifact pairs", acc(G, art, 4, lambda g, i, j: abs(agree(g)[i] - agree(g)[j]) <= .05))
print("D. all-fake GT-no-lift groups: fewer fake-lift frames (diff>=3)")
allfake = [g for g in nolift if all(c["tl"].sum() >= 3 for c in g["cands"])]
show(f"all-fake ({len(allfake)} groups)", acc(allfake, lambda g: -np.array([c["tl"].sum() for c in g["cands"]], float), 3))
print("C. groups with score range > .02")
print(" " * 36 + "|" + "".join(f"{np.mean([np.ptp(g['sc'][k]) > .02 for g in G]):7.2f}" for k in S))
