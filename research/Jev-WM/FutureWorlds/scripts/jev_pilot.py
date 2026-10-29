"""Bounded matched pilot, with fixed data splits and official Decisions audit.

Run from repository root with PYTHONPATH=src. Keys are read without echo.
This is an exploratory experiment, not a statistically powered benchmark.
"""
import argparse
import copy
import getpass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'runs/jev_pilot_20261012'
SOURCE = Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
GPU_PYTHON = '/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python'
SEED = 20261012


def save(path, data):
    from futureworlds.data import write_json
    write_json(path, data)


def load(path):
    return json.loads(Path(path).read_text())


def auth():
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = getpass.getpass('Official API key (hidden): ')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ[key] = 'http://127.0.0.1:17897'
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = 'localhost,127.0.0.1'

    # Experiment-local rate control; keep the training implementation unchanged
    # between arms. Failed attempts also respect this spacing.
    import urllib.error
    from futureworlds.decision_api import DecisionsClient
    original_init = DecisionsClient.__init__
    class PacedOpener:
        def __init__(self, wrapped):
            self.wrapped, self.next_call = wrapped, 0.0
        def open(self, *args, **kwargs):
            delay = self.next_call - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_call = time.monotonic() + 6.0
            try:
                return self.wrapped.open(*args, **kwargs)
            except urllib.error.HTTPError as exc:
                raw = exc.read(4096)
                try:
                    error = json.loads(raw).get('error', {})
                    code = error.get('code')
                except (ValueError, AttributeError):
                    code = 'unparsed'
                print(json.dumps({'event':'api_http_error','status':exc.code,'code':code,
                      'retry_after':exc.headers.get('retry-after')}),flush=True)
                if exc.code == 429:
                    try: retry_after = float(exc.headers.get('retry-after','30'))
                    except ValueError: retry_after = 30
                    self.next_call = time.monotonic() + min(55, max(6,retry_after))
                raise
    def paced_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.opener = PacedOpener(self.opener)
    DecisionsClient.__init__ = paced_init


def prepare():
    from futureworlds.data import read_manifest, read_case
    from futureworlds.checkpoints import file_sha256
    if (EXP / 'protocol.json').exists():
        print('Fixed protocol already exists; not changing it.', flush=True)
        return
    EXP.mkdir(parents=True, exist_ok=True)
    source_train = SOURCE / 'robocasa_data/train_natural.json'
    source_val = SOURCE / 'robocasa_data/val.json'
    train = read_manifest(source_train, 'robocasa')
    val = read_manifest(source_val, 'robocasa')
    assert not ({r['initial_state_id'] for r in train['cases']} & {r['initial_state_id'] for r in val['cases']})
    used = set()
    def select(rows, count):
        rng = np.random.default_rng(SEED)
        rows = [rows[i] for i in rng.permutation(len(rows))]
        result = []
        for i in range(count):
            desired = (i % 2 == 0)
            row = next(r for r in rows if r['initial_state_id'] not in used and bool(r['success']) == desired and r['frames'] >= 10)
            result.append(row); used.add(row['initial_state_id'])
        return result
    choices = {'train': (source_train, select(train['cases'], 20)),
               'calibration': (source_train, select(train['cases'], 4)),
               'validation': (source_val, select(val['cases'], 8))}
    selected = {}
    for split, (source_manifest, rows) in choices.items():
        folder = EXP / 'data' / split
        folder.mkdir(parents=True, exist_ok=True)
        cases = []
        for i, row in enumerate(rows):
            images, actions, text = read_case(source_manifest, row, 52)
            # Deterministic middle window, chosen without image scores or outcomes.
            start = (len(images) - 10) // 2
            dest = folder / f'{i:03d}.npz'
            np.savez_compressed(dest, images=images[start:start+10], actions=actions[start:start+10], instruction=np.array(text))
            cases.append(dict(id=row['id'], episode_id=row['episode_id'], file=dest.name,
                              sha256=file_sha256(dest), frames=10, split='train' if split=='train' else 'val',
                              task=row['task'], initial_state_id=row['initial_state_id'], success=row['success'],
                              source_manifest=str(source_manifest), source_file=row['file'], source_sha256=row['sha256'],
                              source_start=start))
        save(folder/'manifest.json', dict(schema_version=1, dataset='robocasa', cases=cases,
             source_manifest_sha256=file_sha256(source_manifest), selection='seeded unique initial states; balanced outcomes; centered ten-frame window'))
        selected[split] = cases
    schedule = [dict(id=r['id']) for r in selected['train']] * 2
    save(EXP/'schedule.json', schedule)
    original = load(ROOT/'configs/jev-smoke.local.json')
    original.update(train_manifest=str(EXP/'data/train/manifest.json'), evaluation_manifest=str(EXP/'data/validation/manifest.json'),
                    schedule=str(EXP/'schedule.json'), steps=20, checkpoint_every=20, seed=SEED, horizon=4, global_batch=2)
    for arm in ('base', 'hybrid'):
        cfg = copy.deepcopy(original)
        cfg.update(reward='R0' if arm=='base' else 'hybrid', output=str(EXP/arm))
        if arm=='base': cfg.pop('decisions')
        else: cfg['decisions'].update(review_dir=str(EXP/'reviews'), max_api_calls=176)
        save(EXP/f'{arm}.json', cfg)
    save(EXP/'protocol.json', dict(seed=SEED, train_steps=20, global_batch=2, train_exposures=40, schedule_repetitions=2, horizon=4, evaluation_horizons=[4,8],
         beam_width=4, calibration_cases=4, validation_cases=8, train_cases=20, repeats=3,
         selection=selected, evaluation='PSNR/SSIM/LPIPS/MAE/MSE; no API-based model selection',
         hypothesis='Hybrid improves held-out prediction quality versus matched R0; calibration probes judge reliability.',
         calibration='Original, temporal scramble, blackout, and frozen predictions; 3 repeats of originals. Synthetic corruptions are sanity checks, not human annotations.',
         limits='Single seed, 20 updates, 8 independent validation initial states; exploratory, no generalization claim.',
         gpu_memory_cap_fraction=0.25, source_sft_bundle=original['bundle'], source_sft_variant='sft',
         model='gpt-6-luna', rubric='Unchanged from successful single-step smoke test'))
    print(json.dumps({'stage':'prepared', 'experiment':str(EXP), 'counts':{k:len(v) for k,v in selected.items()}}), flush=True)


def calibrate():
    from futureworlds.data import read_case
    from futureworlds.decision_reward import DecisionStore, QUESTIONS
    from futureworlds.decision_api import build_request, normalized_scores, DecisionsClient
    auth()
    client = DecisionsClient(max_calls=30, retries=1)
    store = DecisionStore(dict(review_dir=str(EXP/'calibration/packets'),model='gpt-6-luna',backend='replay'))
    manifest=EXP/'data/calibration/manifest.json'
    rows=load(manifest)['cases']
    results=[]
    for row in rows:
        images, actions, task=read_case(manifest,row,52)
        normal=images[2:]
        scrambled=normal[[0,7,1,6,2,5,3,4]]
        blackout=normal.copy();blackout[3]=0
        frozen=np.repeat(images[1:2],len(normal),axis=0)
        for kind, pred, repeats in [('original',normal,3),('scrambled',scrambled,1),('blackout',blackout,1),('frozen',frozen,1)]:
            key=store.prepare(images[:2],pred,actions[2:],task,'robocasa')
            folder=store.root/key
            packet=load(folder/'request.json')
            with np.load(folder/'frames.npz',allow_pickle=False) as arrays: payload=build_request(packet,arrays)
            for repeat in range(repeats):
                dest=EXP/'calibration/responses'/f'{row["id"]}_{kind}_{repeat}.json'
                if dest.exists():record=load(dest)
                else:
                    response,transport=client.score(payload)
                    scores=normalized_scores(response,QUESTIONS,'gpt-6-luna')
                    record=dict(case=row['id'],kind=kind,repeat=repeat,packet=key,
                                scores=scores,reward=float(np.mean(list(scores.values()))),response=response,transport=transport)
                    save(dest,record)
                results.append(record)
                print(json.dumps({k:record[k] for k in ('case','kind','repeat','reward','scores')}),flush=True)
    paired=[]
    for row in rows:
        records=[r for r in results if r['case']==row['id']]
        original=[r for r in records if r['kind']=='original']
        mean=float(np.mean([r['reward'] for r in original]))
        temporal=float(np.mean([r['scores']['temporal_interaction'] for r in original]))
        paired.append(dict(case=row['id'],original_mean=mean,repeat_range=float(np.ptp([r['reward'] for r in original])),
            **{k+'_delta':mean-next(r['reward'] for r in records if r['kind']==k) for k in ('scrambled','blackout','frozen')},
            scrambled_temporal_delta=temporal-next(r['scores']['temporal_interaction'] for r in records if r['kind']=='scrambled')))
    summary=dict(cases=paired,mean_repeat_range=float(np.mean([r['repeat_range'] for r in paired])),
                 preference_counts={k:sum(r[k+'_delta']>0 for r in paired) for k in ('scrambled','blackout','frozen')},
                 input_tokens=sum(r['response']['usage']['input_tokens'] for r in results),successful_calls=len(results),
                 note='Frozen frames are an ambiguity/reward-hacking probe, not a valid negative without action/dynamics evidence; no human labels were collected.')
    save(EXP/'calibration/summary.json',summary)
    print(json.dumps(summary),flush=True)


def train_arm(arm):
    import torch
    from futureworlds.training import train
    if arm=='hybrid':auth()
    torch.cuda.set_per_process_memory_fraction(0.25,0)
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    try: train(EXP/f'{arm}.json')
    finally:
        save(EXP/arm/'resources.json',dict(seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,gpu=torch.cuda.get_device_name(0),torch=torch.__version__))


def evaluate_models(arms=('sft','base','hybrid')):
    import gc
    import torch
    from futureworlds.exporting import export_model
    from futureworlds.evaluation import evaluate
    torch.cuda.set_per_process_memory_fraction(0.25,0)
    cfg=load(EXP/'base.json')
    for arm in arms:
        bundle=cfg['bundle'];variant='sft'
        if arm!='sft':
            bundle=str(EXP/'bundles'/arm);variant='main'
            export_model(cfg['bundle'],'robocasa',EXP/arm/'checkpoints/step_000020',bundle)
        evaluate(bundle,'robocasa',variant,str(EXP/'data/validation/manifest.json'),str(EXP/'evaluation'/arm),
                 device='cuda',metric_device='cuda',text_model=cfg['text_model'],horizons=[4,8],beam_width=4)
        gc.collect();torch.cuda.empty_cache()
    # Negative control motivated by the frozen-frame calibration failure.
    from futureworlds.evaluation import Metrics
    from futureworlds.data import read_case
    metrics = Metrics('cuda')
    cases = []
    manifest = EXP/'data/validation/manifest.json'
    for row in load(manifest)['cases']:
        images, _, _ = read_case(manifest, row, 52, 8)
        pred = np.repeat(images[1:2].astype(np.float32)/255, 8, axis=0)
        cases.append(dict(id=row['id'], **metrics(pred, images[2:], [4,8])))
    aggregate = {str(h): {k: float(np.mean([r['metrics'][str(h)][k] for r in cases]))
                         for k in cases[0]['metrics'][str(h)]} for h in (4,8)}
    save(EXP/'evaluation/frozen/summary.json',dict(state='COMPLETE',per_case=cases,aggregate=aggregate))


def analyze():
    from futureworlds.checkpoints import file_sha256
    summaries={arm:load(EXP/'evaluation'/arm/'summary.json') for arm in ('sft','base','hybrid')}
    for arm,s in summaries.items():
        assert s['state']=='COMPLETE'
        assert s['protocol']['cases']==summaries['sft']['protocol']['cases']
    bootstrap={}
    rng=np.random.default_rng(SEED)
    draws=rng.integers(0,8,size=(10000,8))
    for h in ('4','8'):
        bootstrap[h]={}
        for metric in ('psnr','ssim','lpips','mae','mse'):
            base=np.array([r['metrics'][h][metric] for r in summaries['base']['per_case']])
            hybrid=np.array([r['metrics'][h][metric] for r in summaries['hybrid']['per_case']])
            diff=hybrid-base
            bootstrap[h][metric]=dict(hybrid_minus_base=float(diff.mean()),
                paired_bootstrap_95=np.quantile(diff[draws].mean(1),[.025,.975]).tolist(),
                hybrid_wins=int(np.sum(diff>0 if metric in ('psnr','ssim') else diff<0)),n=8)
    training={}
    for arm in ('base','hybrid'):
        metrics=[json.loads(line) for line in (EXP/arm/'metrics.jsonl').read_text().splitlines()]
        assert len(metrics)==20 and [r['step'] for r in metrics]==list(range(1,21))
        ck=EXP/arm/'checkpoints/step_000020';receipt=load(ck/'experiment.json')
        for name,sha in receipt['files'].items():assert file_sha256(ck/name)==sha
        training[arm]=dict(steps=len(metrics),resources=load(EXP/arm/'resources.json'),
            checkpoint=str(ck),core_sha256=file_sha256(ck/'model.safetensors'),
            mean_grad_norm=float(np.mean([r['grad_norm'] for r in metrics])),
            ties=sum(r['reward_metrics'].get('decisions/all_tied',0) for r in metrics))
    receipts=[load(p) for p in (EXP/'reviews').glob('*/score.json')]
    tokens=sum(r['official_response']['usage']['input_tokens'] for r in receipts)
    motion = {}
    for arm in ('sft','base','hybrid','frozen'):
        rows = []
        for i, row in enumerate(load(EXP/'data/validation/manifest.json')['cases']):
            path = EXP/'evaluation'/('sft' if arm=='frozen' else arm)/f'case_{i:06d}'/'prediction.npz'
            with np.load(path,allow_pickle=False) as z:
                truth = z['truth'].astype(np.float32)/255
                prev = z['observations'][1:2].astype(np.float32)/255
                pred = np.repeat(prev,len(truth),axis=0) if arm=='frozen' else z['prediction']
            gt_delta = truth - np.concatenate([prev,truth[:-1]],axis=0)
            pred_delta = pred - np.concatenate([prev,pred[:-1]],axis=0)
            rows.append(dict(id=row['id'],delta_mae=float(np.abs(pred_delta-gt_delta).mean()),
                             predicted_motion=float(np.abs(pred_delta).mean()),true_motion=float(np.abs(gt_delta).mean())))
        motion[arm]=dict(per_case=rows,mean_delta_mae=float(np.mean([r['delta_mae'] for r in rows])),
                        mean_predicted_motion=float(np.mean([r['predicted_motion'] for r in rows])),
                        mean_true_motion=float(np.mean([r['true_motion'] for r in rows])))
    report=dict(aggregate={**{arm:s['aggregate'] for arm,s in summaries.items()},
                          'frozen':load(EXP/'evaluation/frozen/summary.json')['aggregate']},
                motion_diagnostics=motion,paired_differences=bootstrap,
                training=training,calibration=load(EXP/'calibration/summary.json'),
                training_api=dict(successful_calls=len(receipts),input_tokens=tokens),
                limitations='Single seed and 8 cases; bootstrap is exploratory, not corrected for multiple comparisons. No human calibration or real robot success test.')
    total=tokens+report['calibration']['input_tokens']
    report['total_api']=dict(input_tokens=total,base_cost_estimate_usd=total*0.10/1e6)
    save(EXP/'comparison.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','calibrate','train-base','train-hybrid','evaluate','evaluate-baselines','analyze']);args=p.parse_args()
    os.chdir(ROOT)
    if args.stage=='prepare':prepare()
    elif args.stage=='calibrate':calibrate()
    elif args.stage.startswith('train-'):train_arm(args.stage.split('-')[1])
    elif args.stage=='evaluate':evaluate_models()
    elif args.stage=='evaluate-baselines':evaluate_models(('sft','base'))
    else:analyze()
