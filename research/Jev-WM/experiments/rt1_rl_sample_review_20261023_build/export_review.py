"""Export saved test predictions for a local, offline qualitative review.

No model inference, GPU use, API scoring, or changes to training artifacts.
Selection is fixed by absolute LPIPS32 reduction from the common SFT baseline.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path("/media/tyshao/Jev-WM/experiments")
QWEN = BASE / "rt1_qwen_a5_h32_20261020"
OPENAI = BASE / "rt1_openai_a5_h32_20261021"
OUT = BASE / "rt1_rl_sample_review_20261023"
RUNS = {
    "sft": QWEN / "evaluation/test/sft",
    "qwen": QWEN / "evaluation/test/rl",
    "openai": OPENAI / "evaluation/test/rl",
}
LABELS = {"gt": "Recorded ground truth", "sft": "RT-1 SFT baseline",
          "qwen": "Qwen3-VL-8B | RL step100", "openai": "OpenAI Decisions | RL step10"}
COLORS = {"gt": "#f1f5f9", "sft": "#a8b3c6", "qwen": "#c4a5ff", "openai": "#70e6c4"}
METRICS = ["lpips", "psnr", "ssim", "mae", "mse"]
FPS = 4


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT18, FONT14, FONT12 = font(18), font(14), font(12)


def load_prediction(key, index, summary):
    p = RUNS[key] / f"case_{index:06d}/prediction.npz"
    if digest(p) != summary["per_case"][index]["prediction_sha256"]:
        raise ValueError(f"Prediction checksum mismatch: {p}")
    with np.load(p, allow_pickle=False) as z:
        prediction = z["prediction"]
        if prediction.shape != (32, 256, 320, 3):
            raise ValueError("Unexpected prediction shape")
        if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
            raise ValueError("Invalid float prediction")
        return dict(pixels=(prediction * 255).round().astype(np.uint8),
                    tokens=z["tokens"][0], truth=z["truth"], observations=z["observations"],
                    source_file=str(p), source_sha256=digest(p))


def render_case(case, summaries):
    index = case["index"]
    folder = OUT / "assets" / f"case_{index:06d}"
    done = folder / "complete.json"
    if done.exists():
        previous = json.loads(done.read_text())
        if previous["id"] == case["id"] and previous["source_hashes"] == case["source_hashes"]:
            print("REUSE", index, flush=True)
            return
        raise ValueError("Existing review export belongs to another prediction")
    frames = folder / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    arrays = {k: load_prediction(k, index, summaries[k]) for k in RUNS}
    gt, obs = arrays["sft"]["truth"], arrays["sft"]["observations"]
    assert gt.shape == (32, 256, 320, 3) and obs.shape == (2, 256, 320, 3)
    for a in arrays.values():
        assert np.array_equal(a["truth"], gt)
        assert np.array_equal(a["observations"], obs)
    # Both experiments must share the exact SFT prediction, not just its metrics.
    other = OPENAI / "evaluation/test/sft" / f"case_{index:06d}/prediction.npz"
    other_summary = json.loads((OPENAI / "evaluation/test/sft/summary.json").read_text())
    assert digest(other) == other_summary["per_case"][index]["prediction_sha256"]
    with np.load(other, allow_pickle=False) as z:
        assert np.array_equal((z["prediction"] * 255).round().astype(np.uint8), arrays["sft"]["pixels"])
        assert np.array_equal(z["truth"], gt) and np.array_equal(z["observations"], obs)
    repeats = {k: [bool(np.array_equal(a["tokens"][t], a["tokens"][t-1])) for t in range(1, 32)]
               for k, a in arrays.items()}
    case["token_repeat_fraction"] = {k: sum(v) / 31 for k, v in repeats.items()}
    order = ["gt", "sft", "qwen", "openai"]
    for t in range(34):
        sheet = Image.new("RGB", (1280, 320), "#111827")
        draw = ImageDraw.Draw(sheet)
        for column, key in enumerate(order):
            x = column * 320
            draw.text((x+8, 5), LABELS[key], font=FONT18, fill=COLORS[key])
            if key == "gt":
                sub = "Recorded future; never supplied during rollout"
            elif key == "sft":
                sub = f"LPIPS32 {case['metrics'][key]['lpips']:.4f} | common baseline"
            else:
                gain = case["gains"][key]["lpips_relative_percent"]
                sub = f"LPIPS32 {case['metrics'][key]['lpips']:.4f} | gain {gain:+.1f}%"
            draw.text((x+8, 28), sub, font=FONT12, fill="#cbd5e1")
            frame = obs[t] if t < 2 else gt[t-2] if key == "gt" else arrays[key]["pixels"][t-2]
            sheet.paste(Image.fromarray(frame), (x, 48))
            phase = f"Shared observation {t+1}/2" if t < 2 else f"Predicted future {t-1}/32"
            if key == "gt" and t >= 2:
                phase = f"Recorded future {t-1}/32"
            if key != "gt" and t >= 3 and repeats[key][t-3]:
                phase += " | repeated tokens"
            draw.text((x+8, 305), phase, font=FONT12, fill="#cbd5e1")
        sheet.save(frames / f"f_{t:02d}.png", compress_level=3)
    shutil.copy2(frames / "f_17.png", folder / "poster.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS),
                    "-i", str(frames / "f_%02d.png"), "-c:v", "libx264", "-threads", "2",
                    "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(folder / "comparison.mp4")], check=True)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json",
                 str(folder / "comparison.mp4")]))["streams"][0]
    assert (probe["width"], probe["height"], probe["nb_frames"], probe["r_frame_rate"]) == (1280, 320, "34", "4/1")
    # Native-resolution contact sheet: four rows and eight timestamps.
    times = [3, 7, 11, 15, 19, 23, 27, 31]
    contact = Image.new("RGB", (2560, 4*288), "#111827")
    draw = ImageDraw.Draw(contact)
    for row, key in enumerate(order):
        y = row * 288
        draw.text((8, y+6), LABELS[key], font=FONT18, fill=COLORS[key])
        for j, t in enumerate(times):
            pixels = gt[t] if key == "gt" else arrays[key]["pixels"][t]
            contact.paste(Image.fromarray(pixels), (320*j, y+32))
            draw.rectangle((320*j, y+32, 320*j+49, y+49), fill="#111827")
            draw.text((320*j+3, y+33), f"f{t+1}", font=FONT12, fill="white")
    contact.save(folder / "contact_sheet.png", compress_level=3)
    contact.resize((1280, 576), Image.Resampling.LANCZOS).save(folder / "contact_preview.jpg", quality=90)
    write(folder / "info.json", case)
    write(done, dict(id=case["id"], source_hashes=case["source_hashes"], video_probe=probe))
    print("EXPORTED", index, case["task"], flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    summaries = {k: json.loads((p / "summary.json").read_text()) for k, p in RUNS.items()}
    ids = [c["id"] for c in summaries["sft"]["per_case"]]
    assert len(ids) == 64
    protocol_keys = ["manifest_sha256", "horizons", "beam_width", "memory", "precision", "tf32",
                     "initial_observations", "gt_refresh", "text_assets"]
    for summary in summaries.values():
        assert [c["id"] for c in summary["per_case"]] == ids
        assert all(summary["protocol"][k] == summaries["sft"]["protocol"][k] for k in protocol_keys)
    manifest_path = BASE / "rt1_memspo200_upstream_20261017/data/test/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = {c["id"]: c for c in manifest["cases"]}
    models = {}
    for key, root, step in [("qwen", QWEN, 100), ("openai", OPENAI, 10)]:
        cp = root / f"train/checkpoints/step_{step:06d}/experiment.json"
        receipt = json.loads(cp.read_text())
        bundle = json.loads((root / "rl_bundle/manifest.json").read_text())
        assert receipt["step"] == step and bundle["training_checkpoint_sha256"] == digest(cp)
        models[key] = dict(steps=step, checkpoint=str(cp.parent), checkpoint_receipt_sha256=digest(cp))
    rows = []
    for index, case_id in enumerate(ids):
        metrics = {k: s["per_case"][index]["metrics"]["32"] for k, s in summaries.items()}
        gains = {}
        for k in ["qwen", "openai"]:
            gains[k] = dict(lpips_absolute=metrics["sft"]["lpips"]-metrics[k]["lpips"],
                lpips_relative_percent=100*(metrics["sft"]["lpips"]-metrics[k]["lpips"])/metrics["sft"]["lpips"],
                psnr_delta_db=metrics[k]["psnr"]-metrics["sft"]["psnr"],
                mae_relative_percent=100*(metrics["sft"]["mae"]-metrics[k]["mae"])/metrics["sft"]["mae"])
        rows.append(dict(index=index, id=case_id, task=tasks[case_id]["instruction"],
            episode_id=tasks[case_id]["episode_id"], start=tasks[case_id]["start"], metrics=metrics,
            gains=gains, per_frame={k: s["per_case"][index]["per_frame"] for k, s in summaries.items()},
            source_hashes={k: s["per_case"][index]["prediction_sha256"] for k, s in summaries.items()},
            asset_dir=f"assets/case_{index:06d}"))
    selections = []
    for key in ["qwen", "openai"]:
        ranked = sorted(rows, key=lambda r: (-r["gains"][key]["lpips_absolute"], r["index"]))
        for rank, row in enumerate(ranked, 1):
            row.setdefault("rank", {})[key] = rank
        for kind, label, row in [("best", "提升第1", ranked[0]), ("best", "提升第2", ranked[1]),
                                 ("best", "提升第3", ranked[2]), ("median", "中位样本", ranked[31]),
                                 ("worst", "退化样本", ranked[-1])]:
            selections.append(dict(model=key, kind=kind, label=label, index=row["index"], rank=row["rank"][key]))
    chosen = sorted({s["index"] for s in selections})
    for index in chosen:
        render_case(rows[index], summaries)
    overview = {}
    for key in ["qwen", "openai"]:
        sft = np.mean([r["metrics"]["sft"]["lpips"] for r in rows])
        rl = np.mean([r["metrics"][key]["lpips"] for r in rows])
        overview[key] = dict(cases=64, improved=sum(r["gains"][key]["lpips_absolute"] > 0 for r in rows),
            regressed=sum(r["gains"][key]["lpips_absolute"] < 0 for r in rows),
            aggregate_lpips_relative_percent=float(100*(sft-rl)/sft))
    payload = dict(schema_version=1, split="test", horizon=32, selection_metric="absolute LPIPS32 reduction from SFT",
        median_definition="rank32 among64, sorted by absolute LPIPS gain", selections=selections,
        cases=[rows[i] for i in chosen], all_cases=rows, overview=overview, models=models,
        playback_fps=FPS, initial_observations=2, predicted_frames=32,
        manifest_sha256=digest(manifest_path), source_summaries={k: dict(path=str(p/"summary.json"),sha256=digest(p/"summary.json")) for k,p in RUNS.items()})
    write(OUT / "selection.json", payload)
    fields = ["index", "id", "task", "qwen_rank", "openai_rank"] + [f"{k}_{m}" for k in RUNS for m in METRICS] + [f"{k}_{m}" for k in ["qwen","openai"] for m in ["lpips_gain","lpips_relative_percent"]]
    with (OUT / "ranking_all64.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            line = {k:row[k] for k in ["index","id","task"]}
            line.update({f"{k}_rank":row["rank"][k] for k in ["qwen","openai"]})
            line.update({f"{k}_{m}":row["metrics"][k][m] for k in RUNS for m in METRICS})
            line.update({f"{k}_lpips_gain":row["gains"][k]["lpips_absolute"] for k in ["qwen","openai"]})
            line.update({f"{k}_lpips_relative_percent":row["gains"][k]["lpips_relative_percent"] for k in ["qwen","openai"]})
            writer.writerow(line)
    template = Path(__file__).with_name("index.template.html").read_text()
    (OUT / "index.html").write_text(template.replace("__REVIEW_JSON__",json.dumps(payload,ensure_ascii=False).replace("</","<\\/")))
    for name in ["viewer.js","styles.css"]:
        shutil.copy2(Path(__file__).with_name(name), OUT / name)
    write(OUT / "files_sha256.json", {p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="files_sha256.json"})
    print("READY",OUT,"selections",len(selections),"unique_cases",len(chosen),flush=True)


if __name__ == "__main__":
    main()
