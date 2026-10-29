"""Categorize every val window by gripper command type and check SFT against a command-only rule."""
import json, numpy as np, collections
from pathlib import Path
R = Path(__file__).resolve().parent
rows = json.load(open(R / "case_table.json"))
for r in rows:
    r["cmd_type"] = "full" if "F" in r["cmd_seq"] and not r["cmd_seq"].rstrip("o").endswith("h") else ("half" if "h" in r["cmd_seq"] else "open")
    if r["gt_any"]:
        r["bucket"] = "gt_lift"
    elif r["success"]:
        r["bucket"] = "success_lift_after_window"
    elif r["cmd_type"] == "half":
        r["bucket"] = "fail_half_close"
    else:
        r["bucket"] = "fail_full_close"
json.dump(rows, open(R / "case_table.json", "w"), indent=1)
c = collections.defaultdict(list)
for r in rows:
    c[r["bucket"]].append(r["pred_any"])
for k, v in c.items():
    print("%-28s n=%3d SFT predicted lift %.2f (%d)" % (k, len(v), np.mean(v), sum(v)))
gt = np.array([r["gt_any"] for r in rows]); pr = np.array([r["pred_any"] for r in rows])
rule = np.array([r["cmd_type"] == "full" for r in rows]).astype(int)
print("command rule 'full close => lift, half => no lift': accuracy %.3f ; SFT %.3f ; SFT agrees with rule %.3f" % (
    (rule == gt).mean(), (pr == gt).mean(), (pr == rule).mean()))
dis = [r for r in rows if (r["cmd_type"] == "full") != bool(r["pred_any"])]
print("SFT departs from command rule on %d windows:" % len(dis))
for r in dis:
    print("  %s %-22s cmd=%s gt=%d sft=%d gt_first=%s pred_first=%s" % (r["id"], r["type"], r["cmd_type"], r["gt_any"], r["pred_any"], r["gt_first"], r["pred_first"]))
print("GT-lift windows: |pred_first - gt_first| median",
      np.median([abs(r["pred_first"] - r["gt_first"]) for r in rows if r["gt_any"] and r["pred_first"] is not None]))
