"""Trajectory reward: per-frame cube (all red pixels) and gripper deviation of a prediction from GT.

No semantic lift/grasp judgement. GT geometry: red mask of the GT frame plus h5 cube/eef poses
projected with the fixed preview camera (cam.py). Prediction geometry: red mask of the predicted
frame; gripper displacement = template match of the GT gripper patch inside the predicted frame.
"""
import numpy as np, cv2
from scipy import ndimage
from cam import project

CAP = 40.0      # px, per-frame error cap
SLACK = 2.0     # px tolerance
PATCH = 24      # half size of gripper template
SEARCH = 40     # px search radius


def red(frame):
    x = frame.astype(np.float32)
    m = (x[..., 0] > 75) & (x[..., 0] > 1.5 * x[..., 1]) & (x[..., 0] > 1.4 * x[..., 2])
    lab, n = ndimage.label(m)
    if n:  # drop specks < 6 px like the logged tracker
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sizes >= 6))
    return m


def cube_error(gm, pm, center, cap=200.0):
    """raw (uncapped<=cap) symmetric red-pixel distances. returns (fwd, bwd, kind, centroid)
    kind: 0 both visible, 1 GT occluded/absent, 2 prediction has no red (vanish), 3 both absent."""
    if not gm.any():
        if not pm.any():
            return 0.0, 0.0, 3, None
        yy, xx = np.nonzero(pm)
        d = np.hypot(xx - center[0], yy - center[1]) - 8  # cube half extent in px
        return float(np.clip(d, 0, cap).mean()), 0.0, 1, np.array([xx.mean(), yy.mean()])
    if not pm.any():
        return 0.0, 0.0, 2, None
    dg = ndimage.distance_transform_edt(~gm)  # distance to nearest GT red pixel
    dp = ndimage.distance_transform_edt(~pm)
    fwd = np.clip(dg[pm] - SLACK, 0, cap).mean()  # ghost / misplaced predicted red
    bwd = np.clip(dp[gm] - SLACK, 0, cap).mean()  # GT cube missing from prediction
    yy, xx = np.nonzero(pm)
    return float(fwd), float(bwd), 0, np.array([xx.mean(), yy.mean()])


def grip_shift(gt_frame, pred_frame, uv):
    """displacement (px) of the GT gripper patch found in pred_frame; red masked to grey."""
    def g(f):
        f = f.copy(); f[red(f)] = 128
        return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gt_frame.shape[:2]
    u, v = int(round(np.clip(uv[0], PATCH, W - PATCH - 1))), int(round(np.clip(uv[1] - 8, PATCH, H - PATCH - 1)))
    tpl = g(gt_frame)[v - PATCH:v + PATCH, u - PATCH:u + PATCH]
    x0, y0 = max(0, u - PATCH - SEARCH), max(0, v - PATCH - SEARCH)
    win = g(pred_frame)[y0:min(H, v + PATCH + SEARCH), x0:min(W, u + PATCH + SEARCH)]
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (bx, by) = cv2.minMaxLoc(r)
    return np.array([bx + x0 - (u - PATCH), by + y0 - (v - PATCH)], float), float(score)


def aggregate(e):
    a = np.asarray(e, float)
    return float(.5 * a.mean() + .5 * np.sort(a)[-max(1, int(np.ceil(len(a) / 4))):].mean())


def gt_geometry(obj_xyz, eef_xyz, reference):
    return dict(masks=[red(f) for f in reference], cube=project(obj_xyz), grip=project(eef_xyz))


def errors(geo, reference, prediction):
    out = dict(fwd=[], bwd=[], kind=[], grip=[], rel=[], match=[])
    for k, (gf, pf) in enumerate(zip(reference, prediction)):
        fwd, bwd, kind, pc = cube_error(geo["masks"][k], red(pf), geo["cube"][k])
        dv, sc = grip_shift(gf, pf, geo["grip"][k])
        if pc is None:
            er = np.nan
        else:  # cube position relative to the gripper, prediction vs GT
            gc = geo["cube"][k] if not geo["masks"][k].any() else np.array(np.nonzero(geo["masks"][k])[::-1]).mean(1)
            er = float(np.hypot(*((pc - gc) - dv)))
        for n, v in (("fwd", fwd), ("bwd", bwd), ("kind", kind), ("grip", float(np.hypot(*dv))), ("rel", er), ("match", sc)):
            out[n].append(v)
    return {k: np.array(v) for k, v in out.items()}


def frame_cost(err, cap=40.0, vanish=12.0, w=(1.0, 0.0, 0.0)):
    """per-frame cost (px): cube term with constant vanish penalty, gripper shift, relative position."""
    cube = np.where(err["kind"] == 2, vanish, .5 * (err["fwd"] + err["bwd"]))
    rel = np.where(np.isnan(err["rel"]), cube, err["rel"])
    return np.minimum(cap, w[0] * np.minimum(cap, cube) + w[1] * np.minimum(cap, err["grip"]) + w[2] * np.minimum(cap, rel))


def score(err, cap=40.0, vanish=12.0, w=(1.0, 0.0, 0.0)):
    """reward in [0,1]"""
    return 1 - aggregate(frame_cost(err, cap, vanish, w)) / cap
