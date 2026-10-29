import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['check', 'all'], default='check', nargs='?')
    parser.add_argument('--gpus', type=int, choices=[2, 4, 8, 16], default=8)
    args = parser.parse_args()
    cfg = read(ROOT / 'train.json')
    audit = read(ROOT / 'recipe_audit.json')
    recipe = read(SOURCE / 'configs/train/rt1_memspo.json')
    for key, value in audit['training_hyperparameters'].items():
        assert cfg[key] == value == recipe[key], key
    for name, expected in read(ROOT / 'source_hashes.json').items():
        assert digest(SOURCE / name) == expected, name
    sys.path.insert(0, str(SOURCE / 'src'))
    from futureworlds.data import read_manifest, disjoint
    train = read_manifest(cfg['train_manifest'], 'rt1')
    test = read_manifest(cfg['evaluation_manifest'], 'rt1')
    disjoint(train, test)
    schedule = read(cfg['schedule'])
    assert len(schedule) == cfg['steps'] * cfg['global_batch']
    assert {x['id'] for x in schedule} <= {x['id'] for x in train['cases']}
    from futureworlds.checkpoints import verify_bundle
    verify_bundle(cfg['bundle'])
    info = {'source_commit': audit['commit'], 'recipe_matches': True, 'gpus_requested': args.gpus, 'accumulation_per_gpu': cfg['global_batch'] // args.gpus, 'training_steps': cfg['steps'], 'training_horizon': cfg['horizon'], 'evaluation_horizons': [10, 20, 32], 'schedule_entries': len(schedule), 'train_episodes': len(train['cases']), 'test_episodes': len(test['cases'])}
    print(json.dumps(info, indent=2), flush=True)
    if args.stage == 'check':
        write(ROOT / 'preparation_check.json', info)
        return
    import torch
    assert torch.cuda.device_count() >= args.gpus, f'Need {args.gpus} visible GPUs; found {torch.cuda.device_count()}. Single-GPU fallback is disabled.'
    write(ROOT / 'launch.json', {**info, 'pid': os.getpid(), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    env = {**os.environ, 'PYTHONPATH': str(SOURCE / 'src'), 'USE_TF': '0', 'USE_FLAX': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'}
    def command(stage, params, distributed=True):
        prefix = [sys.executable, '-u']
        if distributed:
            prefix += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={args.gpus}']
        cmd = prefix + ['-m', 'futureworlds'] + [str(x) for x in params]
        write(ROOT / 'status.json', {'stage': stage, 'pid': os.getpid(), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': cmd})
        print('STAGE', stage, flush=True)
        with (ROOT / 'logs' / (stage + '.log')).open('a') as log:
            subprocess.run(cmd, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    command('cache', ['cache', '--bundle', cfg['bundle'], '--dataset', 'rt1', '--manifest', cfg['train_manifest'], '--output', cfg['token_cache'], '--device', 'cuda', '--text-model', cfg['text_model']])
    if not (ROOT / 'smoke/train/complete.json').exists():
        command('distributed_smoke', ['train', '--config', ROOT / 'smoke.json'])
    # Native trainer creates independent actor, optimizer and frozen reference
    # from SFT for the formal run; smoke weights are never used here.
    if not (ROOT / 'train/complete.json').exists():
        candidates = sorted((ROOT / 'train/checkpoints').glob('step_*/experiment.json'))
        config = ROOT / 'train.json'
        if candidates:
            cfg['resume'] = str(candidates[-1].parent)
            write(ROOT / 'resume.json', cfg)
            config = ROOT / 'resume.json'
        command('train', ['train', '--config', config])
    command('export', ['export-model', '--bundle', cfg['bundle'], '--dataset', 'rt1', '--checkpoint', ROOT / 'train/checkpoints/step_000200', '--output', ROOT / 'rl_bundle'], distributed=False)
    for split in ['validation', 'test']:
        for label, bundle, variant in [('sft', cfg['bundle'], 'sft'), ('rl', ROOT / 'rl_bundle', 'main')]:
            command(f'evaluate_{split}_{label}', ['evaluate', '--bundle', bundle, '--dataset', 'rt1', '--variant', variant, '--manifest', ROOT / f'data/{split}/manifest.json', '--output', ROOT / f'evaluation/{split}/{label}', '--device', 'cuda', '--text-model', cfg['text_model'], '--horizons', 10, 20, 32, '--beam-width', 4])
    subprocess.run([sys.executable, str(ROOT / 'report.py')], cwd=ROOT, check=True, env=env)
    write(ROOT / 'status.json', {'stage': 'COMPLETE', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})

if __name__ == '__main__':
    lock = (ROOT / '.launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except Exception as error:
        write(ROOT / 'status.json', {'stage': 'FAILED', 'error': repr(error), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
