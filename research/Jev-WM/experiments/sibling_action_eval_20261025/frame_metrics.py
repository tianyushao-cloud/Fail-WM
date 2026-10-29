"""Frame metrics (LPIPS-VGG / PSNR / SSIM, same Metrics class as futureworlds.evaluation) for sibling results.

Prediction = saved uint8 *_pred.npz (32 frames after the two real context frames); truth = case frames 2..33.
Usage: frame_metrics.py SPLIT LABEL [LABEL ...]   -> results/<label>/frame_metrics.json (resumable per label)
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path("/media/tyshao/Jev-WM/experiments/sibling_action_eval_20261025")
sys.path.insert(0, "/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_8gpu_20261025/source/src")
from futureworlds.data import read_case
from futureworlds.evaluation import Metrics

DATA = "/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data"
EXTRA = str(R / "data_extra")
split, labels = sys.argv[1], sys.argv[2:]
MANP = f"{EXTRA if split.endswith('_extra') else DATA}/{split}.json"
rows = json.loads(Path(MANP).read_text())["cases"]
m = Metrics("cuda")
for label in labels:
    out = R / "results" / label / "frame_metrics.json"
    if out.exists():
        print(label, "done already"); continue
    prov = json.loads((R / "results" / label / "provenance.json").read_text())
    assert prov["manifest"] == MANP, (prov["manifest"], MANP)
    res = {}
    for r in rows:
        images, _, _ = read_case(MANP, r, 52, 32)
        pred = np.load(R / "results" / label / f"{r['id']}_pred.npz")["prediction"].astype(np.float32) / 255
        res[r["id"]] = m(pred, images[2:34], [32])["per_frame"]
    out.write_text(json.dumps(res))
    print(label, len(res), "cases", flush=True)
