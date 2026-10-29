import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path('/media/tyshao/Jev-WM/experiments/rt1_memspo_20261017')
BUNDLE = '/media/tyshao/Jev-WM/paper_assets/futureworlds_rt1'
TEXT = '/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base'
VENDOR = Path('/media/tyshao/Jev-WM/FutureWorlds/vendor/scholar_futureworlds')

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def status(stage, **extra):
    value = {'stage': stage, 'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid': os.getpid(), **extra}
    write(ROOT / 'status.json', value)
    print(json.dumps(value), flush=True)

def configure():
    locked = read(VENDOR / 'SOURCE_LOCK.json')['files']
    patches = read(VENDOR / 'COMPATIBILITY_PATCHES.json')['files']
    for name, expected in locked.items():
        if name in patches:
            expected = patches[name]['patched_sha256']
        assert digest(VENDOR / name) == expected, f'Source lock mismatch: {name}'
    audit = read(ROOT / 'data_audit.json')
    protocol = {'dataset': 'rt1', 'method': 'memspo', 'seed': 20261017, 'steps': 200, 'global_batch': 16, 'gpus': 1, 'gradient_accumulation': 16, 'horizon': 10, 'lr': 1e-6, 'reward': 'R0=-mean(MAE+LPIPS_VGG)', 'clip': 0.2, 'kl_coef': 0.001, 'diversity_penalty': 0.1, 'candidates': '4 groups x 2 beams; 4 returned candidate videos', 'checkpoint_every': 10, 'comparison_checkpoint': 'fixed final step200; no test-based or validation-based checkpoint selection', 'evaluation': {'horizons': [10, 20, 32], 'beam_width': 4, 'memory': 'full (anchor1+recent6)', 'precision': 'FP32', 'tf32': False, 'initial_observations': 2, 'ground_truth_refresh': False}, 'primary_metric': 'test mean LPIPS at 32 generated frames; report absolute and relative change plus paired episode bootstrap 95% CI', 'other_metrics': ['psnr', 'ssim', 'mae', 'mse'], 'split_counts': audit['split_counts'], 'manifest_sha256': audit['manifest_sha256'], 'bundle_manifest_sha256': digest(Path(BUNDLE) / 'manifest.json'), 'source_commit': read(VENDOR / 'SOURCE_LOCK.json')['commit'], 'source_lock_verified': True, 'paper_reproduction': False, 'limitations': ['One RL seed and limited data', 'Evaluation episodes excluded from this RL run; overlap with original SFT training is unknown', 'Prediction metrics do not measure robot task success', 'Short horizon10 training is compared at horizons10/20/32']}
    target = ROOT / 'protocol.json'
    if target.exists():
        assert read(target) == protocol, 'Protocol changed; use a new experiment directory'
    else:
        write(target, protocol)
    cfg = {'dataset': 'rt1', 'method': 'memspo', 'bundle': BUNDLE, 'text_model': TEXT, 'initial_variant': 'sft', 'device': 'cuda', 'seed': 20261017, 'steps': 200, 'global_batch': 16, 'lr': 1e-6, 'horizon': 10, 'reward': 'R0', 'clip': 0.2, 'kl_coef': 0.001, 'diversity_penalty': 0.1, 'checkpoint_every': 10, 'paper_locked': False, 'cpu_threads': 2, 'train_manifest': str(ROOT / 'data/train/manifest.json'), 'evaluation_manifest': str(ROOT / 'data/test/manifest.json'), 'output': str(ROOT / 'train')}
    write(ROOT / 'train.json', cfg)
    smoke = {**cfg, 'steps': 1, 'global_batch': 1, 'checkpoint_every': 1, 'output': str(ROOT / 'smoke/train'), 'train_manifest': str(ROOT / 'data/smoke_manifest.json')}
    write(ROOT / 'smoke.json', smoke)
    return cfg

def command(stage, args):
    status(stage, command=args)
    with (ROOT / 'logs' / (stage + '.log')).open('a') as log:
        subprocess.run([sys.executable, '-u', '-m', 'futureworlds', *args], cwd=VENDOR, stdout=log, stderr=subprocess.STDOUT, check=True, env={**os.environ, 'PYTHONPATH': str(VENDOR / 'src'), 'USE_TF': '0', 'USE_FLAX': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'})

def evaluation(stage, bundle, variant, split, smoke=False):
    output = ROOT / ('smoke/evaluation' if smoke else f'evaluation/{split}/{variant if variant == "sft" else "rl"}')
    manifest = ROOT / ('data/eval_smoke_manifest.json' if smoke else f'data/{split}/manifest.json')
    # Re-enter completed evaluations: the native evaluator validates every saved
    # case fingerprint and prediction checksum before reusing results.
    command(stage, ['evaluate', '--bundle', str(bundle), '--dataset', 'rt1', '--variant', variant, '--manifest', str(manifest), '--output', str(output), '--device', 'cuda', '--text-model', TEXT, '--horizons', '10', '20', '32'])

def summarize():
    report = {'protocol': read(ROOT / 'protocol.json'), 'checkpoint': 200, 'splits': {}}
    table = []
    for split in ['validation', 'test']:
        sft = read(ROOT / f'evaluation/{split}/sft/summary.json')
        rl = read(ROOT / f'evaluation/{split}/rl/summary.json')
        assert sft['protocol']['cases'] == rl['protocol']['cases']
        for key in ['manifest_sha256', 'horizons', 'beam_width', 'memory', 'metric_device', 'generation_device', 'precision', 'tf32', 'initial_observations', 'gt_refresh', 'implementation', 'text_assets']:
            assert sft['protocol'][key] == rl['protocol'][key], key
        rows = {}
        for horizon in ['10', '20', '32']:
            values = {}
            for metric in ['lpips', 'psnr', 'ssim', 'mae', 'mse']:
                a = np.array([c['metrics'][horizon][metric] for c in sft['per_case']])
                b = np.array([c['metrics'][horizon][metric] for c in rl['per_case']])
                delta = b - a
                rng = np.random.default_rng(20261017)
                boot = delta[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
                lower_better = metric in ['lpips', 'mae', 'mse']
                improvement = (float(a.mean() - b.mean()) if lower_better else float(b.mean() - a.mean()))
                values[metric] = {'sft': float(a.mean()), 'rl': float(b.mean()), 'rl_minus_sft': float(delta.mean()), 'relative_improvement_percent': 100 * improvement / float(a.mean()), 'paired_episode_bootstrap_95ci_rl_minus_sft': np.quantile(boot, [0.025, 0.975]).tolist(), 'episodes': len(a)}
                table.append(f'| {split} | {horizon} | {metric} | {a.mean():.6f} | {b.mean():.6f} | {delta.mean():+.6f} | {values[metric]["relative_improvement_percent"]:+.2f}% |')
            rows[horizon] = values
        report['splits'][split] = rows
    write(ROOT / 'comparison.json', report)
    primary = report['splits']['test']['32']['lpips']
    ci = primary['paired_episode_bootstrap_95ci_rl_minus_sft']
    text = '# RT-1 SFT → MemSPO200 对照\n\n从发布 SFT50k 初始化，200 步、global batch16、horizon10、R0，单 A100 梯度累积。固定最终200步，无测试集选模。\n\n'
    text += f'测试集32帧 LPIPS：SFT {primary["sft"]:.6f} → RL {primary["rl"]:.6f}；相对改善 {primary["relative_improvement_percent"]:+.2f}%；配对 episode bootstrap 95% CI（RL−SFT）[{ci[0]:.6f}, {ci[1]:.6f}]。\n\n'
    text += '| 划分 | 帧数 | 指标 | SFT | RL | RL−SFT | 相对改善 |\n|---|---:|---|---:|---:|---:|---:|\n' + '\n'.join(table)
    text += '\n\n正的相对改善表示指标方向变好。区间跨零时不声称稳健提升。一次训练种子；测试 episode 与本次RL隔离，但原SFT训练集重合未知。图像预测指标不等于机器人成功率。\n'
    (ROOT / 'REPORT.md').write_text(text)
    return primary

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['configure', 'smoke', 'all', 'report'])
    args = p.parse_args()
    if args.stage == 'report':
        summarize()
        return
    cfg = configure()
    if args.stage == 'configure':
        return
    if args.stage == 'smoke':
        if not (ROOT / 'smoke/train/complete.json').exists():
            command('smoke_train', ['train', '--config', str(ROOT / 'smoke.json')])
        evaluation('smoke_evaluate', BUNDLE, 'sft', 'validation', smoke=True)
        ckpt = ROOT / 'smoke/train/checkpoints/step_000001'
        assert digest(ckpt / 'model.safetensors') != digest(Path(BUNDLE) / 'rt1/models/sft/model.safetensors')
        metric = json.loads((ROOT / 'smoke/train/metrics.jsonl').read_text().splitlines()[-1])
        write(ROOT / 'smoke/verification.json', {'training_update_completed': True, 'weights_changed': True, 'evaluation32_completed': True, 'training_metrics': metric, 'rough_training_hours_at_batch16': metric['seconds'] * 16 * 200 / 3600, 'note': 'Estimate from batch1 smoke; excludes checkpoint writes and evaluation; formal batch16 throughput pending'})
        status('SMOKE_COMPLETE')
        return
    assert (ROOT / 'smoke/verification.json').exists(), 'Run smoke first'
    # The checkpoint choice is fixed before evaluation. Training first lets the
    # scarce GPU start useful updates promptly; SFT and RL are then evaluated
    # on exactly the same frozen cohorts without changing the training plan.
    if not (ROOT / 'train/complete.json').exists():
        checkpoints = sorted((ROOT / 'train/checkpoints').glob('step_*/experiment.json'))
        if checkpoints:
            cfg['resume'] = str(checkpoints[-1].parent)
            write(ROOT / 'resume.json', cfg)
            config = ROOT / 'resume.json'
        else:
            config = ROOT / 'train.json'
        command('train', ['train', '--config', str(config)])
    command('export', ['export-model', '--bundle', BUNDLE, '--dataset', 'rt1', '--checkpoint', str(ROOT / 'train/checkpoints/step_000200'), '--output', str(ROOT / 'rl_bundle')])
    for split in ['validation', 'test']:
        evaluation(f'eval_{split}_sft', BUNDLE, 'sft', split)
        evaluation(f'eval_{split}_rl', ROOT / 'rl_bundle', 'main', split)
    primary = summarize()
    status('COMPLETE', test_lpips32=primary)

if __name__ == '__main__':
    lock = (ROOT / '.runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except Exception as exc:
        status('FAILED', error=repr(exc))
        raise
