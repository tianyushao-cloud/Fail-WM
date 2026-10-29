"""Exact stall profile by future-frame position (no judge)."""
import json, sys
from pathlib import Path
import numpy as np
folder = Path(__file__).resolve().parent / "cases" / sys.argv[1]
files = sorted(folder.glob("[0-9][0-9][0-9].npz"))
stall, static, moved_all = [], [], []
for f in files:
    with np.load(f) as z:
        init, ref, preds = z["initial"], z["reference_raw"], z["predictions"]
    prev_r = np.concatenate([init[-1:], ref[:-1]])
    moved = np.abs(ref.astype(np.float32) - prev_r.astype(np.float32)).mean((1, 2, 3)) > 1.0
    moved_all.append(moved)
    for p in preds:
        prev = np.concatenate([init[-1:], p[:-1]])
        st = np.array([np.array_equal(a, b) for a, b in zip(p, prev)])
        static.append(st); stall.append(st & moved)
static, stall, moved_all = map(np.array, (static, stall, moved_all))
T = static.shape[1]
print(f"windows {len(files)}  candidates {len(static)}  horizon {T}")
for s in range(0, T, 4):
    e = min(s + 4, T)
    print(f"future frames {s+1:2d}-{e:2d}: candidate static {static[:, s:e].mean():.2f}  GT moving {moved_all[:, s:e].mean():.2f}  stall(static & GT moving) {stall[:, s:e].mean():.2f}")
for h in (10, 16, 24, 32):
    if h <= T:
        run = [max((len(x) for x in "".join("1" if v else "0" for v in c[:h]).split("0")), default=0) for c in static]
        print(f"first {h:2d} frames: static fraction {static[:, :h].mean():.2f}; candidates with a frozen run >= 5 frames: {np.mean(np.array(run) >= 5):.2f}")
