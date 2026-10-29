"""Pure official Decisions reward versus fixed SFT; bounded and resumable pilot.

Run with PYTHONPATH=src, USE_TF=0, USE_FLAX=0. API key is read without echo.
Exact content-addressed official receipts may be reused; no manual labels.
"""
import argparse
import gc
import getpass
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'runs/jev_pilot_20261012'
EXP = ROOT / 'runs/jev_pure_20261012'


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    from futureworlds.data import write_json
    write_json(path, value)


def prepare():
    if (EXP / 'protocol.json').exists():
        return
    EXP.mkdir(parents=True, exist_ok=True)
    cfg = load(PRIOR / 'hybrid.json')
    cfg.update(reward='decisions', output=str(EXP / 'train'), checkpoint_every=1)
    for key in ('base_reward', 'base_scale', 'weight'):
        cfg['decisions'].pop(key, None)
    cfg['decisions'].update(review_dir=str(EXP / 'reviews'), retries=0, max_api_calls=160)
    copied = []
    for source in sorted((PRIOR / 'reviews').glob('*/score.json')):
        receipt = load(source)
        if 'official_response' not in receipt:
            raise ValueError('Only official API receipts may be reused')
        shutil.copytree(source.parent, EXP / 'reviews' / source.parent.name)
        copied.append(source.parent.name)
    save(EXP / 'config.json', cfg)
    save(EXP / 'protocol.json', dict(
        reward='decisions', reward_formula='mean(object_consistency, temporal_interaction)',
        no_reconstruction_reward=True, unchanged_kl_regularization=cfg['kl_coef'],
        initial_variant='sft', target_steps=cfg['steps'], checkpoint_every=1,
        seed=cfg['seed'], global_batch=cfg['global_batch'], horizon=cfg['horizon'],
        validation_manifest=cfg['evaluation_manifest'], evaluation_horizons=[4, 8], beam_width=4,
        prior_sft_evaluation=str(PRIOR / 'evaluation/sft'),
        copied_official_receipts=copied,
        stop_rule='20 updates or first API/compute/training failure; evaluate last saved update, never select by validation score',
        limitations='Single seed, eight held-out initial states. Rubric unchanged; known static-frame bias. No action-consistency rubric or human preference labels.',
        gpu_memory_fraction=.20))
    print(json.dumps({'event':'prepared', 'copied_official_receipts':len(copied), 'target_steps':cfg['steps']}), flush=True)


def train_run(resume=False):
    import torch
    from futureworlds.training import train
    from futureworlds.decision_api import DecisionsClient
    prepare()
    cfg_path = EXP / 'config.json'
    cfg = load(cfg_path)
    if resume:
        checkpoints = sorted((EXP / 'train/checkpoints').glob('step_*'))
        if not checkpoints:
            raise ValueError('No completed checkpoint to resume')
        cfg['resume'] = str(checkpoints[-1])
        cfg_path = EXP / 'resume.json'
        save(cfg_path, cfg)
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = getpass.getpass('Official API key (hidden): ')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ[key] = 'http://127.0.0.1:17897'
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = 'localhost,127.0.0.1'
    network = {'attempts':0, 'errors':[]}
    original_init = DecisionsClient.__init__
    class PacedOpener:
        def __init__(self, wrapped):
            self.wrapped, self.next_call = wrapped, 0.0
        def open(self, *args, **kwargs):
            time.sleep(max(0, self.next_call - time.monotonic()))
            self.next_call = time.monotonic() + 6
            network['attempts'] += 1
            try:
                return self.wrapped.open(*args, **kwargs)
            except urllib.error.HTTPError as exc:
                try:
                    code = json.loads(exc.read(4096)).get('error', {}).get('code')
                except (ValueError, AttributeError):
                    code = 'unparsed'
                # Preserve quota evidence, never credentials or response text.
                event = dict(status=exc.code, code=code, headers={
                    k:v for k,v in exc.headers.items()
                    if k.lower().startswith('x-ratelimit-') or k.lower() == 'retry-after'})
                network['errors'].append(event)
                print(json.dumps({'event':'api_error', **event}), flush=True)
                raise
    def paced_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.opener = PacedOpener(self.opener)
    DecisionsClient.__init__ = paced_init
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    status = {'state':'COMPLETE', 'target_steps':cfg['steps']}
    try:
        train(cfg_path)
    except Exception as exc:
        status.update(state='STOPPED', error_type=type(exc).__name__, error=str(exc))
        print(json.dumps(status), flush=True)
    finally:
        checkpoints = sorted((EXP / 'train/checkpoints').glob('step_*'))
        status.update(completed_steps=int(checkpoints[-1].name.split('_')[-1]) if checkpoints else 0,
                      checkpoint=str(checkpoints[-1]) if checkpoints else None,
                      network=network, seconds=time.monotonic()-started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        save(EXP / 'status.json', status)
        print(json.dumps(status), flush=True)
    return status


def evaluate_run():
    import torch
    from futureworlds.exporting import export_model
    from futureworlds.evaluation import evaluate
    from futureworlds.checkpoints import file_sha256
    status, cfg = load(EXP / 'status.json'), load(EXP / 'config.json')
    if not status['completed_steps']:
        print('No trained checkpoint available; no comparison fabricated.', flush=True)
        return
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    name = 'decisions_' + str(status['completed_steps']) + 'step'
    bundle = EXP / 'bundles' / name
    if not (bundle / 'manifest.json').exists():
        export_model(cfg['bundle'], cfg['dataset'], status['checkpoint'], bundle)
    # Reuse deterministic SFT evaluation only after verifying its exact protocol/assets.
    sft = load(PRIOR / 'evaluation/sft/summary.json')
    protocol = sft['protocol']
    assert sft['state'] == 'COMPLETE'
    assert protocol['manifest_sha256'] == file_sha256(cfg['evaluation_manifest'])
    assert protocol['bundle_manifest_sha256'] == file_sha256(Path(cfg['bundle']) / 'manifest.json')
    assert protocol['variant'] == 'sft' and protocol['horizons'] == [4, 8]
    assert protocol['beam_width'] == 4 and protocol['gt_refresh'] is False
    for i, case in enumerate(sft['per_case']):
        assert file_sha256(PRIOR / 'evaluation/sft' / f'case_{i:06d}/prediction.npz') == case['prediction_sha256']
    evaluate(str(bundle), cfg['dataset'], 'main', cfg['evaluation_manifest'], str(EXP / 'evaluation' / name),
             device='cuda', metric_device='cuda', text_model=cfg['text_model'], horizons=[4, 8], beam_width=4)
    save(EXP / 'evaluation_paths.json', dict(sft=str(PRIOR / 'evaluation/sft'), decisions=str(EXP / 'evaluation' / name)))
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'train': train_run(args.resume)
    else: evaluate_run()
