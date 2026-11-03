#!/usr/bin/env python3
"""Download the pinned private 1000-episode dataset, verify it, optionally restore paths."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def safe(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Path escapes destination: " + relative)
    return target

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--restore-root", type=Path, help="Optional original-tree root, e.g. /media/tyshao. Requires additional disk space.")
    args = parser.parse_args()
    pin = json.loads((Path(__file__).resolve().parents[1] / "migration/dataset_release.json").read_text())
    snapshot_download(repo_id=pin["repo_id"], repo_type="dataset", revision=pin["revision"], local_dir=args.output)
    manifest_path = args.output / "data_manifest.json"
    if sha(manifest_path) != pin["manifest_sha256"]:
        raise ValueError("Dataset manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    for i, row in enumerate(manifest["files"]):
        src = safe(args.output, row["path"])
        if src.stat().st_size != row["bytes"] or sha(src) != row["sha256"]:
            raise ValueError("Dataset checksum mismatch: " + row["path"])
        if args.restore_root:
            dst = safe(args.restore_root, row["restore_path"])
            if dst.exists():
                if sha(dst) != row["sha256"]:
                    raise FileExistsError("Refusing to overwrite different file: " + str(dst))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                if sha(dst) != row["sha256"]:
                    raise ValueError("Restored checksum mismatch: " + str(dst))
        if (i + 1) % 1000 == 0:
            print("Verified", i + 1, "files", flush=True)
    print("Verified all", len(manifest["files"]), "payload files;", manifest["episodes"], "episodes")

if __name__ == "__main__":
    main()
