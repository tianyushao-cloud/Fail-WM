#!/usr/bin/env python3
"""Create a verified FutureWorlds bundle with an adapted RoboCasa codec."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def install_bundle(base, codec, output):
    base, codec, output = map(Path, (base, codec, output))
    if output.exists():
        raise FileExistsError(output)
    if not (base / "manifest.json").is_file():
        raise FileNotFoundError(base / "manifest.json")
    if not (codec / "diffusion_pytorch_model.safetensors").is_file():
        raise FileNotFoundError(codec / "diffusion_pytorch_model.safetensors")
    shutil.copytree(base, output)
    target = output / "robocasa" / "codec"
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        shutil.copyfile(codec / name, target / name)
    manifest = json.loads((output / "manifest.json").read_text())
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": path.relative_to(output).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest["files"] = files
    manifest["robocasa_codec_source"] = str(codec)
    manifest["robocasa_codec_sha256"] = sha256(target / "diffusion_pytorch_model.safetensors")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "codec_sha256": manifest["robocasa_codec_sha256"],
                      "files": len(files)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", required=True)
    parser.add_argument("--codec-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    install_bundle(args.base_bundle, args.codec_checkpoint, args.output)


if __name__ == "__main__":
    main()
