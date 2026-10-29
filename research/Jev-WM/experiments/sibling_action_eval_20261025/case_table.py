"""Per-case table for the SFT sibling report: GT outcome, SFT prediction, gripper closure."""
import json, h5py, numpy as np, collections
from pathlib import Path
R = Path(__file__).resolve().parent
MAN = {r["id"]: r for r in json.load(open("/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/val.json"))["cases"]}
rows = []
for f in sorted((R / "results/sft").glob("g*_b*.json")):
    d = json.loads(f.read_text())
    rise = np.array([np.nan if v is None else v for v in d["pred_rise_px"]], float)
    pl = np.where(np.isnan(rise), 0, rise > 8).astype(int)
    gt = np.array(d["gt_lift"])
    with h5py.File(MAN[d["id"]]["source_h5"]) as h:
        g = h["observations/gripper_joint_position"][:]
        fl = h["events/flags"][:]
    i = (np.arange(32) + 2) * 4
    width = (g[i, 0] - g[i, 1]) * 100
    rows.append(dict(id=d["id"], group=d["group"], type=d["failure_type"] or "success", success=d["success"],
        gt_any=int(gt.any()), pred_any=int(pl.sum() >= 3), gt_lift=gt.tolist(), pred_lift=pl.tolist(),
        gt_first=int(np.argmax(gt)) if gt.any() else None, pred_first=int(np.argmax(pl)) if pl.any() else None,
        pred_rise_max=float(np.nanmax(rise)) if (~np.isnan(rise)).any() else None,
        contact_any=int(fl[i, 0].any()), final_width_cm=round(float(width[-1]), 2),
        closed=int(width.min() < 7.5), missing=float(np.isnan(rise).mean())))
json.dump(rows, open(R / "case_table.json", "w"), indent=1)
print("GT no-lift windows:")
for r in rows:
    if r["gt_any"] == 0:
        print("  %s %-22s contact=%d width=%4.1fcm closed=%d -> SFT %s rise_max=%s" % (
            r["id"], r["type"], r["contact_any"], r["final_width_cm"], r["closed"],
            "LIFT" if r["pred_any"] else "no-lift", r["pred_rise_max"]))
c = collections.defaultdict(list)
for r in rows:
    if r["gt_any"] == 0:
        c[("contact" if r["contact_any"] else "no-contact", "closed" if r["closed"] else "open")].append(r["pred_any"])
for k, v in sorted(c.items()):
    print(k, "optimism %.2f n=%d" % (np.mean(v), len(v)))
