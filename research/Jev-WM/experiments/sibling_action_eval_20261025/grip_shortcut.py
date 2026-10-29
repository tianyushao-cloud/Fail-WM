"""Is SFT's lift decision explained by the gripper command alone?

Gripper command (dim 6 of each 13-d control block): -1 open, 0 half close, +1 full close.
What matters is the command the episode settles on, so use the last nonzero-change value
in the window, plus the final finger width from the simulator.
"""
import json, numpy as np, collections
from pathlib import Path
R = Path(__file__).resolve().parent
rows = json.load(open(R / "case_table.json"))
for r in rows:
    d = json.load(open(R / "results/sft" / f"{r['id']}.json"))
    g = np.array(d["grip_cmd"]).ravel()
    r["cmd_final"] = int(round(g[-1]))
    r["cmd_seq"] = "".join({-1: "o", 0: "h", 1: "F"}[int(round(x))] for x in g[::4])
    r["width_class"] = "closed_on_nothing" if r["final_width_cm"] < 1 else ("partial" if r["final_width_cm"] < 7.5 else "still_open")
json.dump(rows, open(R / "case_table.json", "w"), indent=1)
for key in ("cmd_final", "width_class"):
    c = collections.defaultdict(list)
    for r in rows:
        c[(r[key], r["gt_any"])].append(r["pred_any"])
    print("(%s, GT lift) -> SFT predicted-lift rate" % key)
    for k, v in sorted(c.items(), key=str):
        print("  %-18s gt_lift=%d  pred_lift %.2f  n=%d" % (k[0], k[1], np.mean(v), len(v)))
print("GT no-lift examples, command sequence (every 4th frame):")
for r in rows:
    if r["gt_any"] == 0:
        print("  %s %-22s %s width=%4.1f -> %s" % (r["id"], r["type"], r["cmd_seq"], r["final_width_cm"], "LIFT" if r["pred_any"] else "-"))
