"""Pure trajectory reward: per-frame 2D deviation of the predicted cube and gripper from GT.

No lift/grasp semantics and no VLM. Offline evaluation: experiments/traj_reward_20261027
(REPORT.md, variant T_full). GT geometry = red mask of the GT frame + h5 cube/eef poses
projected with the fixed preview camera (reference frame k = h5 index (2+k)*pack).
  cube     symmetric mean distance between predicted and GT red pixels (all blobs, so ghost
           cubes count); GT occluded -> distance to the projected cube centre; prediction
           without red -> constant `vanish` px
  gripper  displacement of the GT gripper patch found in the predicted frame (normalized
           cross-correlation, same formula as cv2.TM_CCOEFF_NORMED)
  relative (pred cube - pred gripper) vs (GT cube - GT gripper)
frame cost = weighted sum (each term and the sum capped at `cap` px);
reward = 1 - (0.5 mean + 0.5 mean of worst quarter of frames) / cap.
"""
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from scipy import ndimage, signal
from .batch import TensorBatch
from .rewards import CodecWorker
from .robocasa_object_reward import rule_scores
from .data import write_json

CAM_POS = np.array([.9, -1.4, 1.9])
_x = np.array([.83205, .5547, 0]); _y = np.array([-.2691, .40365, .8745])
_x /= np.linalg.norm(_x); _y /= np.linalg.norm(_y)
CAM_R = np.stack([_x, _y, np.cross(_x, _y)])
CAM_F = 240 / np.tan(np.deg2rad(24))  # fovy 48 at 640x480
SLACK = 2.0     # px tolerance in the cube distance
CUBE_HALF = 8   # px, cube half extent when GT is occluded
PATCH = 24      # half size of the gripper template
SEARCH = 40     # px search radius


def project(p):
    """world xyz (...,3) -> packed 320x256 pixel (...,2)"""
    c = (np.asarray(p) - CAM_POS) @ CAM_R.T
    return np.stack([(320 + CAM_F * c[..., 0] / -c[..., 2]) * .5, (240 - CAM_F * c[..., 1] / -c[..., 2]) * 256 / 480], -1)


def red(frame):
    x = frame.astype(np.float32)
    m = (x[..., 0] > 75) & (x[..., 0] > 1.5 * x[..., 1]) & (x[..., 0] > 1.4 * x[..., 2])
    lab, n = ndimage.label(m)
    if n:  # drop specks < 6 px like the rule tracker
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sizes >= 6))
    return m


def cube_error(gm, pm, center, cap=200.0):
    """(fwd, bwd, kind, centroid); kind 0 both visible, 1 GT occluded, 2 prediction has no red, 3 both absent"""
    if not gm.any():
        if not pm.any():
            return 0.0, 0.0, 3, None
        yy, xx = np.nonzero(pm)
        d = np.hypot(xx - center[0], yy - center[1]) - CUBE_HALF
        return float(np.clip(d, 0, cap).mean()), 0.0, 1, np.array([xx.mean(), yy.mean()])
    if not pm.any():
        return 0.0, 0.0, 2, None
    dg = ndimage.distance_transform_edt(~gm)
    dp = ndimage.distance_transform_edt(~pm)
    fwd = np.clip(dg[pm] - SLACK, 0, cap).mean()  # ghost / misplaced predicted red
    bwd = np.clip(dp[gm] - SLACK, 0, cap).mean()  # GT cube missing from prediction
    yy, xx = np.nonzero(pm)
    return float(fwd), float(bwd), 0, np.array([xx.mean(), yy.mean()])


def gray(f):
    f = f.copy(); f[red(f)] = 128
    return (.299 * f[..., 0] + .587 * f[..., 1] + .114 * f[..., 2]).astype(np.float64)


def match(win, tpl):
    """argmax (x, y) and score of TM_CCOEFF_NORMED"""
    t = tpl - tpl.mean(); th, tw = t.shape; n = th * tw
    num = signal.correlate(win, t, mode="valid", method="fft")
    box = np.ones((th, tw))
    s1 = signal.correlate(win, box, mode="valid", method="fft")
    s2 = signal.correlate(win * win, box, mode="valid", method="fft")
    den = np.sqrt(np.maximum(s2 - s1 * s1 / n, 0) * (t * t).sum())
    r = np.where(den > 1e-6 * max(1.0, den.max()), num / np.maximum(den, 1e-12), 0.0)
    by, bx = np.unravel_index(np.argmax(r), r.shape)
    return int(bx), int(by), float(r[by, bx])


def grip_shift(gt_gray, pred_gray, uv):
    H, W = gt_gray.shape
    u, v = int(round(np.clip(uv[0], PATCH, W - PATCH - 1))), int(round(np.clip(uv[1] - 8, PATCH, H - PATCH - 1)))
    tpl = gt_gray[v - PATCH:v + PATCH, u - PATCH:u + PATCH]
    x0, y0 = max(0, u - PATCH - SEARCH), max(0, v - PATCH - SEARCH)
    win = pred_gray[y0:min(H, v + PATCH + SEARCH), x0:min(W, u + PATCH + SEARCH)]
    bx, by, score = match(win, tpl)
    return np.array([bx + x0 - (u - PATCH), by + y0 - (v - PATCH)], float), score


def errors(geo, prediction):
    out = dict(fwd=[], bwd=[], kind=[], grip=[], rel=[], match=[])
    for k, pf in enumerate(prediction):
        gm = geo["masks"][k]
        fwd, bwd, kind, pc = cube_error(gm, red(pf), geo["cube"][k])
        dv, sc = grip_shift(geo["gray"][k], gray(pf), geo["grip"][k])
        if pc is None:
            er = np.nan
        else:
            gc = geo["cube"][k] if not gm.any() else np.array(np.nonzero(gm)[::-1]).mean(1)
            er = float(np.hypot(*((pc - gc) - dv)))
        for n, x in (("fwd", fwd), ("bwd", bwd), ("kind", kind), ("grip", float(np.hypot(*dv))), ("rel", er), ("match", sc)):
            out[n].append(x)
    return {k: np.array(v) for k, v in out.items()}


def aggregate(e):
    a = np.asarray(e, float)
    return float(.5 * a.mean() + .5 * np.sort(a)[-max(1, int(np.ceil(len(a) / 4))):].mean())


def frame_cost(err, cap, vanish, w):
    cube = np.where(err["kind"] == 2, vanish, .5 * (err["fwd"] + err["bwd"]))
    rel = np.where(np.isnan(err["rel"]), cube, err["rel"])
    return np.minimum(cap, w[0] * np.minimum(cap, cube) + w[1] * np.minimum(cap, err["grip"]) + w[2] * np.minimum(cap, rel))


def score(err, cap, vanish, w):
    return 1 - aggregate(frame_cost(err, cap, vanish, w)) / cap


class RoboCasaTrajReward:
    def __init__(self, codec, config, device, horizon):
        self.worker = CodecWorker(codec, None)
        self.config = config
        self.horizon = horizon
        self.root = Path(config["log_dir"])
        self.weights = config["weights"]
        if list(self.weights) != ["cube", "gripper", "relative"]:
            raise ValueError("weights must be cube/gripper/relative")
        self.w = tuple(float(self.weights[k]) for k in ("cube", "gripper", "relative"))
        self.cap, self.vanish = float(config["cap"]), float(config["vanish"])
        self.schedule = json.loads(Path(config["schedule"]).read_text())
        self.cases = {r["id"]: r for r in json.loads(Path(config["train_manifest"]).read_text())["cases"]}

    def poses(self, cursor):
        row = self.cases[self.schedule[cursor]["id"]]
        with h5py.File(row["source_h5"], "r") as f:
            obj = f["observations/object_pose_wxyz"][:, :3]
            eef = f["observations/eef_executed13"][:, :3]
            lift = f["events/flags"][:, 2]
        index = (np.arange(self.horizon) + 2) * row.get("pack", 4)
        if index[-1] >= len(obj):
            raise ValueError(f"{row['id']}: h5 shorter than window")
        return obj[index], eef[index], lift[index].astype(int), row["id"]

    @torch.no_grad()
    def __call__(self, frames, context, actions, pixels, *, task, cursor, seed):
        begin = time.monotonic()
        codes = torch.stack([f["response"] for f in frames], 1)
        n = codes.shape[0]
        if codes.shape[1] != self.horizon:
            raise ValueError("Unexpected horizon")
        pred = self.worker.detokenize(
            TensorBatch(dict(tokens=codes, ctx_tokens=context.expand(n, -1, -1))), None
        ).batch["pixels"][:, 1:]

        def rgb(x):
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite decoded pixels")
            return x.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()

        initial = rgb(pixels[0, :2])
        reference = rgb(pixels[0, 2:2 + self.horizon])
        predictions = np.stack([rgb(x) for x in pred])
        del pred
        obj, eef, lift, case_id = self.poses(cursor)
        geo = dict(masks=[red(f) for f in reference], gray=[gray(f) for f in reference], cube=project(obj), grip=project(eef))
        errs = [errors(geo, p) for p in predictions]
        axes = dict(cube=np.array([score(e, self.cap, self.vanish, (1, 0, 0)) for e in errs]),
                    gripper=np.array([score(e, self.cap, self.vanish, (0, 1, 0)) for e in errs]),
                    relative=np.array([score(e, self.cap, self.vanish, (0, 0, 1)) for e in errs]))
        reward = np.array([score(e, self.cap, self.vanish, self.w) for e in errs])
        # Monitoring only (not in the reward): legacy tracker axes and fake-lift frames vs h5 lift.
        scored = [rule_scores(reference, p) for p in predictions]
        y0 = next((t["xy"][1] for t in scored[0][1]["gt"] if t["xy"] is not None), None)
        rise = np.array([[0 if y0 is None or t["xy"] is None else int(y0 - t["xy"][1] > 8) for t in s[1]["pred"]] for s in scored])
        fake_lift = ((rise == 1) & (lift[None] == 0)).sum(1)
        raw = reward.copy()
        if np.ptp(reward) < self.config.get("minimum_group_spread", 0.02):
            reward[:] = reward.mean()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.config.get("save_clips", True):
            np.savez_compressed(self.root / f"{cursor:06d}_clips.npz", initial=initial, reference=reference,
                                predictions=predictions, tokens=codes.cpu().numpy())
        write_json(self.root / f"{cursor:06d}.json", dict(
            cursor=cursor, seed=seed, id=case_id, task=str(task), weights=self.weights, cap=self.cap, vanish=self.vanish,
            axes={k: v.tolist() for k, v in axes.items()}, gt_lift=lift.tolist(), raw_reward=raw.tolist(),
            reward=reward.tolist(), fake_lift_frames=fake_lift.tolist(),
            legacy={k: [s[0][k] for s in scored] for k in ("position", "persistence")},
            errors=[{k: np.where(np.isnan(v), -1, v).round(3).tolist() for k, v in e.items()} for e in errs],
            seconds=time.monotonic() - begin))
        stats = dict(judged_groups=1, informative_groups=int(np.ptp(reward) > 1e-6),
                     judge_seconds=time.monotonic() - begin, reward_spread=float(np.ptp(raw)),
                     gt_lift_windows=int(lift.any()),
                     fake_lift_frames=float(fake_lift.mean()) if not lift.any() else 0.0,
                     **{f"score_sum_{k}": float(v.mean()) for k, v in axes.items()})
        return torch.tensor(reward, dtype=torch.float32, device=codes.device), stats
