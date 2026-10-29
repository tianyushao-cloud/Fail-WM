"""Stage A report (CPU): lifted-call rate per simulator category and separability, per question variant."""
import collections, json, sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
rows = [json.loads(l) for l in (OUT / "results" / "stage_a.jsonl").read_text().splitlines() if l.strip()]
CATS = ["free_open", "free_closed", "contact_nolift", "grasp_nolift", "lifted"]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    by[r["variant"]][r["cat"]].append(r["p_lift"])
print("fraction of frames called LIFTED (p_lift > 0.5); mean p_lift in brackets")
print(f"{'variant':8s}" + "".join(f"{c:>22s}" for c in CATS) + "   AUC lifted vs non-lifted (closed/contact/grasp)")
for v in sorted(by):
    line = f"{v:8s}"
    for c in CATS:
        x = np.asarray(by[v][c])
        line += f"{(x > .5).mean():12.2f} ({x.mean():.2f}) n={len(x):3d}" if len(x) else f"{'-':>22s}"
    hard = by[v]["free_closed"] + by[v]["contact_nolift"] + by[v]["grasp_nolift"]
    line += f"   {auc(by[v]['lifted'], hard):.3f}"
    print(line)
print("\nfull answer distribution (mean probability per digit)")
for v in sorted(by):
    for c in CATS:
        P = np.array([r["p"] for r in rows if r["variant"] == v and r["cat"] == c])
        if len(P):
            print(f"  {v} {c:15s} " + " ".join(f"{x:.2f}" for x in P.mean(0)))
