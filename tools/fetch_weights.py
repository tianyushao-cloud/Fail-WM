#!/usr/bin/env python3
"""Download the exact private HF snapshot linked to this code release."""
import argparse,hashlib,json
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--revision');a=ap.parse_args()
 release_path=Path(__file__).resolve().parents[1]/'migration/hf_release.json'
 if not release_path.exists():raise SystemExit('The verified HF release pin has not been committed yet; update this code checkout after the migration finishes.')
 release=json.loads(release_path.read_text())
 revision=a.revision or release['revision']
 snapshot_download(repo_id=release['repo_id'],repo_type='model',revision=revision,local_dir=a.output)
 p=a.output/'weights_manifest.json'
 if not a.revision and hashlib.sha256(p.read_bytes()).hexdigest()!=release['weights_manifest_sha256']:raise RuntimeError('Downloaded manifest does not match the recorded release')
 print(f'Downloaded {release["repo_id"]}@{revision} to {a.output}; run restore_project.py to verify weight bytes and restore original paths.')
if __name__=='__main__':main()
