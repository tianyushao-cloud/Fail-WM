"""Sanity check the pixel-rise 'lift' call: for GT no-lift windows, how high does the GT cube itself rise in the image?"""
import json, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent
rows = {r["id"]: r for r in json.load(open(R / "case_table.json"))}
for cid, r in rows.items():
    if r["gt_any"]:
        continue
    d = json.load(open(R / "results/sft" / f"{cid}.json"))
    g = np.array([np.nan if v is None else v for v in d["gt_rise_px"]], float)
    p = np.array([np.nan if v is None else v for v in d["pred_rise_px"]], float)
    print("%s %-22s GT rise max %6.1f  pred rise max %6.1f  pred>8px frames %2d  -> %s" % (
        cid, r["type"], np.nanmax(g), np.nanmax(p), int(np.nansum(p > 8)), "LIFT" if r["pred_any"] else "-"))
