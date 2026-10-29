"""Sibling-action test: does the WM's predicted outcome follow the action it is given?

val.json has 10 initial states x 10 trajectories. Siblings share the same first two
frames and differ only in actions, each with a simulator outcome (h5 lift flag).
Every model gets the real first two frames + that trajectory's own actions, horizon 32,
beam 4, full memory, seed 0 (same protocol as evaluation_step20_val12).

Usage: python evaluate.py --label sft
       python evaluate.py --label rl30 --checkpoint .../train/checkpoints/step_000030
       python evaluate.py --label sft_s1 --seed 1   (repeat with another sampling seed)
Resumable: finished cases are skipped. Analysis: python report.py
"""
import os, sys, json, time, argparse, traceback
from pathlib import Path

R = Path(__file__).resolve().parent
SRC = Path("/media/tyshao/Jev-WM/experiments/robocasa_state_memspo_g8_3gpu_20261025/source/src")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0",
                  OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, str(SRC))
import numpy as np, torch, h5py
from futureworlds.data import read_case, write_json
from futureworlds.pipeline import WorldPipeline
from futureworlds.exporting import export_model
from futureworlds.checkpoints import load_world_model
from futureworlds.robocasa_object_reward import track

BUNDLE = "/media/tyshao/fail_WM/FutureWorlds_SFT/previews_wm_codec34k_20261012/natural_bundle"
DATA = "/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data"
# failure70 val/test episodes not in val/test.json (same initial states); built by build_extra.py
EXTRA = "/media/tyshao/Jev-WM/experiments/sibling_action_eval_20261025/data_extra"
TEXT = "/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base"
H = 32


def ys(clip):
    return np.array([np.nan if t["xy"] is None else t["xy"][1] for t in track(clip)])


def jl(x):
    return [None if np.isnan(v) else float(v) for v in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", default=None, help="RL checkpoint dir; omit for the SFT bundle")
    ap.add_argument("--memory_fraction", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0, help="no effect: beam decoding is deterministic")
    ap.add_argument("--split", choices=["val", "test", "val_extra", "test_extra"], default="val")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    MANIFEST = f"{EXTRA if a.split.endswith('_extra') else DATA}/{a.split}.json"
    out = R / "results" / a.label
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(MANIFEST).read_text())["cases"]
    pipe = WorldPipeline(BUNDLE, "robocasa", "sft", "cuda", TEXT)
    if a.checkpoint:
        exported = R / "bundles" / a.label
        export_model(BUNDLE, "robocasa", a.checkpoint, exported)
        pipe.model = load_world_model(exported, "robocasa", "main", "cuda", verify=False)
    write_json(out / "provenance.json", dict(label=a.label, split=a.split, checkpoint=a.checkpoint, bundle=BUNDLE, manifest=MANIFEST,
                                             horizon=H, beam=4, memory="full", seed=a.seed))
    for i, row in enumerate(rows):
        dest = out / f"{row['id']}.json"
        if dest.exists():
            continue
        t0 = time.monotonic()
        images, actions, text = read_case(MANIFEST, row, 52, H)
        torch.manual_seed(a.seed)
        with torch.inference_mode():
            pred = pipe.predict(images[:2], actions, text, H, 4, "full")["prediction"]
        rgb = np.rint(pred * 255).clip(0, 255).astype(np.uint8)
        with h5py.File(row["source_h5"], "r") as f:
            flags = f["events/flags"][:]
        idx = (np.arange(H) + 2) * row.get("pack", 4)
        y0 = ys(images[1:2])[0]
        np.savez_compressed(out / f"{row['id']}_pred.npz", prediction=rgb)
        write_json(dest, dict(
            id=row["id"], group=row["initial_state_id"], success=row["success"], failure_type=row["failure_type"],
            gt_lift=flags[idx, 2].astype(int).tolist(), gt_grasp=flags[idx, 1].astype(int).tolist(),
            gt_rise_px=jl(y0 - ys(images[2:2 + H])), pred_rise_px=jl(y0 - ys(rgb)),
            grip_cmd=actions[2:2 + H][:, [6, 19, 32, 45]].tolist(), seconds=time.monotonic() - t0))
        print(json.dumps(dict(event="done", n=i + 1, of=len(rows), id=row["id"], s=round(time.monotonic() - t0, 1))), flush=True)
    write_json(out / "complete.json", dict(n=len(rows)))
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
