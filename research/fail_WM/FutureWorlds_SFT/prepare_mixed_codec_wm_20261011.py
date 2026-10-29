#!/usr/bin/env python3
"""Freeze an audited codec and prepare the original two WM SFT protocols."""
import argparse
import json
from pathlib import Path

ROOT = Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
EXP = ROOT / 'experiments_mixed_codec_wm_20261011'
AUDIT = ROOT / 'evals_codec_current_20261011/report.json'
RUN = ROOT / 'runs/paper_adapt_mixed_codec_20261009'

def load(p):
    return json.loads(Path(p).read_text())

def frozen(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if load(p) != value:
            raise ValueError(f'Frozen record differs: {p}')
    else:
        p.write_text(json.dumps(value, indent=2) + '\n')

def prepare():
    from futureworlds.checkpoints import file_sha256, verify_bundle
    from install_robocasa_codec_bundle import install_bundle
    audit = load(AUDIT)
    reference = load(RUN / 'reference_codec_validation.json')
    validations = {}
    for line in (RUN / 'metrics.jsonl').read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # The running trainer can be appending its last line.
        if 'validation' in r:
            validations[r['step']] = r['validation']
    score = lambda v: v['failure']['roi_mae_255'] + 32 * v['failure']['miss_rate']
    candidates = []
    for label, result in audit['codecs'].items():
        if not label.startswith('mixed_'):
            continue
        step = int(label.split('_')[-1])
        v = validations[step]
        summary = result['summary']
        uf, sc, rf = (summary[k] for k in ('unlifted_failure', 'success_control', 'release_failure'))
        numeric_ok = (v['success']['roi_mae_255'] <= reference['success']['roi_mae_255'] * 1.10 + .25
                      and v['success']['miss_rate'] <= reference['success']['miss_rate'] + .01
                      and score(v) < score(reference))
        event_ok = (uf['n'] == 24 and uf['lifted'] <= 1 and uf['undetected'] == 0
                    and sc['n'] == 20 and sc['lifted'] == 20
                    and sum(rf.values()) == 7 and rf['vanished'] == 0
                    and rf['fell_like_gt'] == 7)
        candidates.append(dict(step=step, checkpoint=result['path'], score=score(v),
                               numeric_pass=numeric_ok, event_pass=event_ok,
                               validation=v, events=summary))
    passing = [c for c in candidates if c['numeric_pass'] and c['event_pass']]
    if len(passing) < 2:
        raise ValueError('Need at least two recent checkpoints passing both checks before freezing a stable codec')
    best = min(passing, key=lambda c: (c['score'], c['step']))
    recent = sorted(candidates, key=lambda c: c['step'])
    roi = [c['validation']['failure']['roi_mae_255'] for c in recent]
    if (max(roi) - min(roi)) / max(roi) > .05:
        raise ValueError('Recent failure ROI spread exceeds the 5% operational stability check')
    checkpoint = Path(best['checkpoint'])
    receipt = load(checkpoint / 'experiment.json')
    if any(file_sha256(checkpoint / name) != sha for name, sha in receipt['files'].items()):
        raise ValueError('Checkpoint checksum mismatch')
    bundle = ROOT / f"assets/futureworlds_bundle_mixed_codec_step{best['step']:06d}_20261011"
    base = ROOT / 'assets/futureworlds_bundle_rt1_init'
    verify_bundle(base)
    if not bundle.exists():
        install_bundle(base, checkpoint, bundle)
    verify_bundle(bundle)
    codec_sha = file_sha256(checkpoint / 'diffusion_pytorch_model.safetensors')
    if load(bundle / 'manifest.json')['robocasa_codec_sha256'] != codec_sha:
        raise ValueError('Existing bundle has a different codec')
    selected = dict(selected=best, candidates=candidates, reference_score=score(reference),
                    codec_sha256=codec_sha, bundle=str(bundle), audit_sha256=file_sha256(AUDIT),
                    audit=str(AUDIT), stability_failure_roi_relative_spread=(max(roi)-min(roi))/max(roi),
                    selection_note='Additional independent acceptance, original codec selection/training untouched; 51 validation cases only, no test data')
    frozen(EXP / 'codec_selection.json', selected)
    protocols = {}
    for variant, prefix, count in [('success', 'success_only', 443), ('natural', 'natural', 800)]:
        original = ROOT / f'runs/{prefix}_sft_new_codec_10k.json'
        cfg = load(original)
        cfg.pop('resume', None)
        name = f"{prefix}_sft_mixed_codec{best['step']}_10k_20261011"
        cfg.update(bundle=str(bundle), token_cache=str(ROOT / 'cache/token_cache' / name),
                   output=str(ROOT / 'runs' / name))
        assert cfg['steps'] == 10000 and cfg['global_batch'] == 64 and cfg['micro_batch'] == 8
        assert cfg['checkpoint_every'] == 2000 and cfg['seed'] == 20261005 and 'schedule' not in cfg
        cases = load(cfg['train_manifest'])['cases']
        assert len(cases) == count and sum(bool(r['success']) for r in cases) == 443
        old = load(original)
        old.pop('resume', None)
        changes = {k: [old.get(k), cfg.get(k)] for k in old.keys() | cfg.keys() if old.get(k) != cfg.get(k)}
        assert set(changes) == {'bundle', 'token_cache', 'output'}
        frozen(EXP / f'{variant}.json', cfg)
        protocols[variant] = dict(config=str(EXP / f'{variant}.json'), original=str(original),
                                  config_changes=changes, train_cases=count, train_success=443,
                                  train_failure=count-443, manifest_sha256=file_sha256(cfg['train_manifest']))
    frozen(EXP / 'protocol.json', dict(codec_step=best['step'], codec_sha256=codec_sha, variants=protocols,
                                      initialization='Same original RT1 weights and seed; retrain both WMs, not resume old models',
                                      sampling='Original uniform Schedule; E2 and loss modifications excluded'))
    print(json.dumps({'selected_step':best['step'], 'bundle':str(bundle), 'protocol':str(EXP / 'protocol.json')}, indent=2))

def preflight():
    from futureworlds.checkpoints import file_sha256, verify_bundle
    from futureworlds.data import read_manifest, disjoint
    selected, protocol = load(EXP / 'codec_selection.json'), load(EXP / 'protocol.json')
    if file_sha256(AUDIT) != selected['audit_sha256']:
        raise ValueError('Audit changed after experiment freeze')
    verify_bundle(Path(selected['bundle']))
    if file_sha256(Path(selected['bundle']) / 'robocasa/codec/diffusion_pytorch_model.safetensors') != selected['codec_sha256']:
        raise ValueError('Selected codec changed')
    smoke = load(EXP / 'smoke_result.json')
    if smoke['state'] != 'PASS' or smoke['codec_bundle'] != selected['bundle']:
        raise ValueError('Selected codec must pass cache/SFT smoke')
    for variant, entry in protocol['variants'].items():
        cfg = load(entry['config'])
        old = load(entry['original']); old.pop('resume', None)
        changed = {k: [old.get(k), cfg.get(k)] for k in old.keys() | cfg.keys() if old.get(k) != cfg.get(k)}
        if changed != entry['config_changes']:
            raise ValueError(f'SFT protocol changed: {variant}')
        if file_sha256(cfg['train_manifest']) != entry['manifest_sha256']:
            raise ValueError('Training manifest changed')
        if file_sha256(Path(cfg['initial_checkpoint']) / 'model.safetensors') != cfg['initial_sha256']:
            raise ValueError('RT1 initial checkpoint changed')
        train = read_manifest(cfg['train_manifest'], 'robocasa')
        val = read_manifest(cfg['evaluation_manifest'], 'robocasa')
        disjoint(train, val)
        for key in ('initial_state_id', 'initial_state_sha256'):
            if {r[key] for r in train['cases']} & {r[key] for r in val['cases']}:
                raise ValueError(f'Train/validation initial-state overlap: {key}')
    print('Preflight passed: frozen codec, bundle, RT1, manifests, split separation, and original SFT protocol')

def runtime(variant):
    cfg = load(EXP / f'{variant}.json'); out = Path(cfg['output'])
    if (out / 'complete.json').exists():
        print('complete'); return
    checkpoints = sorted((out / 'checkpoints').glob('step_*/experiment.json'))
    if checkpoints:
        cfg['resume'] = str(checkpoints[-1].parent)
    elif (out / 'run.json').exists():
        # The trainer allows an identical resume fingerprint even before the first
        # checkpoint only if run.json is absent. Do not delete it implicitly.
        raise RuntimeError(f'Interrupted before first checkpoint; inspect {out}')
    (EXP / f'{variant}_runtime.json').write_text(json.dumps(cfg, indent=2) + '\n')
    print('run')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['prepare', 'preflight', 'runtime'])
    p.add_argument('--variant', choices=['success', 'natural'])
    a = p.parse_args()
    if a.phase == 'runtime':
        if a.variant is None: p.error('--variant is required for runtime')
        runtime(a.variant)
    else:
        {'prepare':prepare, 'preflight':preflight}[a.phase]()
