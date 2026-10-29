import base64
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from futureworlds.data_readers.rt1 import records, parse
from futureworlds.data import write_json, read_manifest, read_case, disjoint
from futureworlds.checkpoints import file_sha256

ROOT = Path('/media/tyshao/Jev-WM/experiments/rt1_memspo_20261017')
ASSETS = Path('/media/tyshao/Jev-WM/paper_assets')
RAW = ROOT / 'raw'
SEED = 20261017
TEXT = '/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base'

def download(item):
    name = Path(item['name']).name
    p = RAW / name
    if p.exists() and (p.stat().st_size != int(item['size']) or base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode() != item['md5Hash']):
        p.rename(p.with_name(p.name + '.invalid'))
    if not p.exists():
        old = ASSETS / 'rt1_data_probe' / name
        if old.exists():
            shutil.copyfile(old, p)
        else:
            temp = p.with_name(p.name + '.partial')
            subprocess.run(['curl', '--proxy', 'http://192.168.32.28:18000', '--fail', '--location', '--silent', '--show-error', '--retry', '4', '--connect-timeout', '25', '--max-time', '900', '--continue-at', '-', '--output', str(temp), 'https://storage.googleapis.com/gresearch/' + item['name']], check=True)
            temp.rename(p)
    assert p.stat().st_size == int(item['size']), name
    h = hashlib.md5()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    assert base64.b64encode(h.digest()).decode() == item['md5Hash'], name
    print('RAW_VERIFIED', name, flush=True)
    return p

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'data_audit.json').exists():
        for split in ['train', 'validation', 'test']:
            path = ROOT / 'data' / split / 'manifest.json'
            m = read_manifest(path, 'rt1')
            for row in m['cases']:
                read_case(path, row, 13, 32)
        print('EXISTING_DATA_VERIFIED', flush=True)
        return
    RAW.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((ASSETS / 'audit_rt1/rt1_official_object_inventory.json').read_text())
    objects = sorted([x for x in inventory if '.tfrecord-' in x['name']], key=lambda x: x['name'])[:16]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        shards = list(pool.map(download, objects))
    tokenizer = AutoTokenizer.from_pretrained(TEXT, local_files_only=True)
    eligible, excluded, seen = [], [], set()
    for shard in shards:
        for index, offset, record in records(shard):
            eid = hashlib.sha256(record).hexdigest()
            if eid in seen:
                excluded.append({'episode_id': eid, 'reason': 'duplicate_record'})
                continue
            seen.add(eid)
            jpeg, actions, text = parse(record)
            if len(jpeg) < 34 or len(tokenizer(text)['input_ids']) > 64:
                excluded.append({'episode_id': eid, 'reason': 'too_short_or_text_over_64_tokens', 'frames': len(jpeg)})
                continue
            eligible.append({'episode_id': eid, 'path': str(shard), 'offset': offset, 'length': len(record), 'frames': len(jpeg), 'instruction': text})
    eligible.sort(key=lambda x: hashlib.sha256(f'{SEED}:split:{x["episode_id"]}'.encode()).hexdigest())
    assert len(eligible) >= 224, f'Only {len(eligible)} eligible episodes; need at least 128 train + 96 held out'
    selected = {'validation': eligible[:32], 'test': eligible[32:96], 'train': eligible[96:608]}
    manifests = {}
    for split, episodes in selected.items():
        dest = ROOT / 'data' / split
        dest.mkdir(parents=True, exist_ok=True)
        cases = []
        for row in episodes:
            with open(row['path'], 'rb') as f:
                f.seek(row['offset'])
                raw = f.read(row['length'])
            assert hashlib.sha256(raw).hexdigest() == row['episode_id']
            jpeg, actions, text = parse(raw)
            start = int(hashlib.sha256(f'{SEED}:window:{row["episode_id"]}'.encode()).hexdigest(), 16) % (len(jpeg) - 33)
            images = np.stack([np.array(Image.open(io.BytesIO(b)).convert('RGB')) for b in jpeg[start:start + 34]])
            assert images.shape == (34, 256, 320, 3)
            actions = actions[start:start + 34]
            assert actions.shape == (34, 13) and np.isfinite(actions).all()
            file = dest / f'{len(cases):06d}.npz'
            np.savez_compressed(file, images=images, actions=actions.astype(np.float32), instruction=np.array(text))
            cases.append({'id': f'{row["episode_id"]}_{start:04d}', 'episode_id': row['episode_id'], 'split': split, 'file': file.name, 'sha256': file_sha256(file), 'frames': 34, 'start': start, 'source_sha256': row['episode_id'], 'source_path': row['path'], 'source_offset': row['offset'], 'source_length': row['length'], 'instruction': text})
        manifest = {'schema_version': 1, 'dataset': 'rt1', 'cases': cases, 'paper_cohort': False, 'action_alignment': 'actions[t]: frame t -> frame t+1', 'seed': SEED}
        write_json(dest / 'manifest.json', manifest)
        manifests[split] = manifest
        print('PREPARED', split, len(cases), flush=True)
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        disjoint(manifests[a], manifests[b])
    write_json(ROOT / 'data_audit.json', {'seed': SEED, 'source': 'Official gresearch fractal20220817_data 0.1.0, first 16 sorted TFRecord shards', 'source_objects': objects, 'records': len(seen), 'eligible': len(eligible), 'excluded': excluded, 'split_counts': {k: len(v['cases']) for k, v in manifests.items()}, 'split_rule': 'Sort eligible unique record SHA256 IDs by SHA256(seed:split:episode_id); first32 validation, next64 test, next up to512 train', 'window_rule': 'One deterministic uniformly indexed legal 34-frame window per episode; start=SHA256(seed:window:episode_id) mod (episode_frames-33)', 'episode_disjoint': True, 'original_sft_overlap': 'Unknown; original SFT training split is unpublished', 'manifest_sha256': {k: file_sha256(ROOT / 'data' / k / 'manifest.json') for k in manifests}})
    write_json(ROOT / 'data/smoke_manifest.json', {**manifests['train'], 'cases': [{**manifests['train']['cases'][0], 'file': 'train/' + manifests['train']['cases'][0]['file']}]})
    write_json(ROOT / 'data/eval_smoke_manifest.json', {**manifests['validation'], 'cases': [{**manifests['validation']['cases'][0], 'file': 'validation/' + manifests['validation']['cases'][0]['file']}]})

if __name__ == '__main__':
    main()
