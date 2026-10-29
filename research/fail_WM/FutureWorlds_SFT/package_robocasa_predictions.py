#!/usr/bin/env python3
"""Export both full-rollout prediction sets as reviewable MP4 files and one ZIP."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path("/media/tyshao/fail_WM/FutureWorlds_SFT")
EXPORT = ROOT / "generated_video_export"
ARCHIVE = ROOT / "generated_videos_all_models.zip"
RUNS = (
    ("success_only", "success_step10000_full"),
    ("success_failure", "natural_step10000_full"),
)
FPS = 5


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_frames(path):
    with np.load(path, allow_pickle=False) as data:
        frames = data["prediction"].copy()
    if frames.ndim != 4 or frames.shape[1:] != (256, 320, 3) or len(frames) == 0:
        raise ValueError(f"Unexpected prediction shape: {path}: {frames.shape}")
    if frames.dtype != np.uint8:
        if not np.issubdtype(frames.dtype, np.floating) or not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError(f"Invalid prediction pixels: {path}")
        frames = np.rint(frames * 255).astype(np.uint8)
    return frames


def encoded_frame_count(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def encode(frames, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "320x256", "-r", str(FPS),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp),
    ]
    proc = subprocess.run(command, input=frames.tobytes(), capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed for {path}: {proc.stderr.decode(errors='replace')}")
    if encoded_frame_count(temp) != len(frames):
        raise ValueError(f"Encoded frame count mismatch: {path}")
    temp.replace(path)


def main():
    EXPORT.mkdir(parents=True, exist_ok=True)
    entries = []
    for label, run in RUNS:
        for split in ("val", "test"):
            manifest = json.loads((ROOT / "robocasa_data" / f"{split}.json").read_text())
            if len(manifest["cases"]) != 100:
                raise ValueError(f"Expected 100 {split} cases")
            for index, row in enumerate(manifest["cases"]):
                source = ROOT / "evals_full" / run / split / f"case_{index:06d}" / "prediction.npz"
                receipt = source.with_name("result.json")
                if not source.is_file() or not receipt.is_file():
                    raise FileNotFoundError(source)
                if json.loads(receipt.read_text())["id"] != row["id"]:
                    raise ValueError(f"Receipt ID mismatch: {source}")
                frames = video_frames(source)
                if len(frames) != row["frames"] - 2:
                    raise ValueError(f"Frame count mismatch: {source}")
                relative = Path(label) / split / f"{index:03d}_{row['id']}.mp4"
                target = EXPORT / relative
                if not target.is_file() or encoded_frame_count(target) != len(frames):
                    encode(frames, target)
                entries.append({"model": label, "split": split, "index": index,
                                "episode_id": row["id"], "file": relative.as_posix(),
                                "frames": len(frames), "fps": FPS,
                                "sha256": sha256(target)})
                print(f"{len(entries)}/400 {relative}", flush=True)

    if len(entries) != 400 or len({e["file"] for e in entries}) != 400:
        raise ValueError("Export is incomplete or contains duplicate filenames")
    manifest_path = EXPORT / "manifest.json"
    manifest_path.write_text(json.dumps({
        "description": "Full generated RoboCasa rollouts from the two step-10000 SFT checkpoints",
        "frames_per_second": FPS,
        "resolution": [320, 256],
        "pixel_conversion": "round(clip(prediction_float_0_to_1 * 255))",
        "encoding": "H.264, yuv420p, CRF 14",
        "note": "Videos contain predicted future frames only; the two observed input frames are omitted. No success labels are included.",
        "videos": entries,
    }, indent=2) + "\n")
    temp_archive = ARCHIVE.with_suffix(".partial.zip")
    with zipfile.ZipFile(temp_archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        bundle.write(manifest_path, "manifest.json")
        for entry in entries:
            bundle.write(EXPORT / entry["file"], entry["file"])
    with zipfile.ZipFile(temp_archive) as bundle:
        if len(bundle.namelist()) != 401 or bundle.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    temp_archive.replace(ARCHIVE)
    print(json.dumps({"archive": str(ARCHIVE), "videos": len(entries),
                      "bytes": ARCHIVE.stat().st_size, "sha256": sha256(ARCHIVE)}), flush=True)


if __name__ == "__main__":
    main()
