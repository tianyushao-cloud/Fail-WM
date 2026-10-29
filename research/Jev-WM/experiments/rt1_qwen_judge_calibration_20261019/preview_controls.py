import sys, numpy as np
from pathlib import Path
from PIL import Image
from calibrate_axes import controls
from qwen_axes import _full_strip
ROOT = Path(__file__).resolve().parent
f = ROOT / "cases" / sys.argv[1]
c = {k: v for k, v in np.load(f / "000.npz").items()}
o = {k: v for k, v in np.load(f / "001.npz").items()}
ctrl = controls(c, o)
rows = [("gt_codec", c["reference_codec"])] + list(ctrl.items())
ts = [3, 9, 15, 17, 21, 25, 31]
sheet = Image.new("RGB", (160 * len(ts), 128 * len(rows)), "white")
for r, (n, clip) in enumerate(rows):
    for j, t in enumerate(ts):
        sheet.paste(Image.fromarray(clip[t]).resize((160, 128)), (160 * j, 128 * r))
out = ROOT / "control_preview"; out.mkdir(exist_ok=True)
sheet.save(out / "controls.png")
_full_strip(c["initial"], ctrl["flicker"]).save(out / "strip_flicker.png")
_full_strip(c["initial"], c["reference_codec"]).save(out / "strip_gt.png")
print([n for n, _ in rows])
