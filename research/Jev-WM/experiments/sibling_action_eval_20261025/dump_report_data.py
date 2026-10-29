"""Dump everything the HTML report needs into one JSON."""
import json, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent
rows = json.load(open(R / "case_table.json"))
by = {r["id"]: r for r in rows}
CASES = ["g00006_b00", "g00006_b03", "g00006_b07", "g00084_b08", "g00084_b09", "g00022_b05", "g00014_b07", "g00059_b03"]
cases = []
for cid in CASES:
    r = by[cid]
    d = json.load(open(R / "results/sft" / f"{cid}.json"))
    rise = [None if v is None else round(v, 1) for v in d["pred_rise_px"]]
    cases.append(dict(id=cid, group=r["group"], type=r["type"], success=r["success"], cmd=r["cmd_type"],
                      cmd_seq=r["cmd_seq"], width=r["final_width_cm"], contact=r["contact_any"],
                      gt=r["gt_lift"], pred=r["pred_lift"], pred_missing=[v is None for v in d["pred_rise_px"]],
                      pred_rise=rise, gt_any=r["gt_any"], pred_any=r["pred_any"]))
    print(cid, r["type"], r["cmd_type"], "GT ", "".join(map(str, r["gt_lift"])))
    print(" " * len(cid), " " * len(r["type"]), " " * len(r["cmd_type"]), "SFT", "".join("." if m else str(p) for p, m in zip(r["pred_lift"], [v is None for v in d["pred_rise_px"]])))
buckets = {}
for r in rows:
    b = r["bucket"]
    ok = r["pred_any"] == r["gt_any"]
    buckets.setdefault(b, []).append(int(ok))
summary = dict(n=len(rows), acc=float(np.mean([r["pred_any"] == r["gt_any"] for r in rows])),
               buckets={k: dict(n=len(v), correct=int(sum(v))) for k, v in buckets.items()},
               missing=float(np.mean([r["missing"] for r in rows])))
print(json.dumps(summary, indent=1))
json.dump(dict(cases=cases, summary=summary), open(R / "report_media/report_data.json", "w"))
