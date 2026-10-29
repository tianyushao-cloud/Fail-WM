"""Paired comparison of pure Decisions RL with unchanged senior SFT weights."""
import json
from pathlib import Path
import numpy as np
from futureworlds.checkpoints import file_sha256
from futureworlds.data import write_json

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'runs/jev_pure_20261012'

def load(path):
    return json.loads(Path(path).read_text())

paths = {k:Path(v) for k,v in load(EXP / 'evaluation_paths.json').items()}
summaries = {k:load(v / 'summary.json') for k,v in paths.items()}
sft, rl = summaries['sft'], summaries['decisions']
for name in ('manifest_sha256', 'horizons', 'beam_width', 'memory', 'precision', 'tf32',
             'initial_observations', 'gt_refresh', 'cases', 'text_assets', 'implementation'):
    assert sft['protocol'][name] == rl['protocol'][name], name
ids = sft['protocol']['cases']
for arm, summary in summaries.items():
    assert summary['state'] == 'COMPLETE'
    assert [row['id'] for row in summary['per_case']] == ids
    for i, row in enumerate(summary['per_case']):
        assert file_sha256(paths[arm] / f'case_{i:06d}/prediction.npz') == row['prediction_sha256']

rng = np.random.default_rng(20261012)
draws = rng.integers(0, len(ids), (10000, len(ids)))
paired = {}
for h in ['4', '8']:
    paired[h] = {}
    for metric in ['lpips', 'mae', 'mse', 'psnr', 'ssim']:
        before = np.array([r['metrics'][h][metric] for r in sft['per_case']])
        after = np.array([r['metrics'][h][metric] for r in rl['per_case']])
        diff = after - before
        paired[h][metric] = dict(sft=float(before.mean()), decisions=float(after.mean()),
            decisions_minus_sft=float(diff.mean()),
            paired_bootstrap_95=np.quantile(diff[draws].mean(1), [.025, .975]).tolist(),
            decisions_wins=int(np.sum(diff > 0 if metric in ('psnr', 'ssim') else diff < 0)), n=len(ids))

motion = {}
for arm, path in paths.items():
    rows = []
    for i, cid in enumerate(ids):
        with np.load(path / f'case_{i:06d}/prediction.npz', allow_pickle=False) as z:
            truth = z['truth'].astype(np.float32)/255
            previous = z['observations'][1:2].astype(np.float32)/255
            pred = z['prediction']
        gt_delta = truth - np.concatenate([previous, truth[:-1]])
        delta = pred - np.concatenate([previous, pred[:-1]])
        mask = np.abs(truth-previous).mean(-1) > .02
        rows.append(dict(id=cid, motion_roi_mae=float((np.abs(pred-truth)*mask[...,None]).sum()/max(1, 3*mask.sum())),
                         predicted_motion=float(np.abs(delta).mean()), true_motion=float(np.abs(gt_delta).mean()),
                         delta_mae=float(np.abs(delta-gt_delta).mean()), roi_fraction=float(mask.mean())))
    motion[arm] = dict(per_case=rows, **{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='id'})

status = load(EXP / 'status.json')
checkpoint = Path(status['checkpoint'])
receipt = load(checkpoint / 'experiment.json')
for name, digest in receipt['files'].items():
    assert file_sha256(checkpoint / name) == digest
metrics = [json.loads(x) for x in (EXP / 'train/metrics.jsonl').read_text().splitlines()]
assert len(metrics) == status['completed_steps']
for record in metrics:
    assert np.isfinite(record['grad_norm'])
    assert all(key.startswith('decisions/') for key in record['reward_metrics'])
old_keys = set(load(EXP / 'protocol.json')['copied_official_receipts'])
new_receipts = [load(p) for p in (EXP / 'reviews').glob('*/score.json') if p.parent.name not in old_keys]
consumed = sorted({p.parent.name for p in (EXP / 'reviews').glob('*/request.json')})
report = dict(state='COMPLETE_COMPARISON', training=status, metrics=paired, motion=motion,
              checkpoint_sha256=file_sha256(checkpoint / 'model.safetensors'),
              successful_new_api_calls=len(new_receipts), new_input_tokens=sum(r['official_response']['usage']['input_tokens'] for r in new_receipts),
              limitations=load(EXP / 'protocol.json')['limitations'],
              evaluation='Unchanged SFT predictions reused after protocol, asset and prediction hash checks. No SFT retraining. Last saved RL checkpoint, no validation-based checkpoint selection.')
write_json(EXP / 'comparison.json', report)
lines = ['# 纯 Decisions RL 与原始 SFT 对比', '',
         '直接使用学长原始 SFT 权重作为起点和基线，没有重新训练 SFT。', '',
         f"训练奖励仅为 OpenAI Decisions 的物体一致性与时序交互评分均值；没有 R0/LPIPS/MAE 奖励。保留原 MemSPO 的 KL 正则。目标 20 步，实际完成 **{status['completed_steps']} 步**；每步保存 checkpoint。", '',
         '固定 8 个独立验证初始状态，Beam=4，预测 4/8 帧，不注入未来真实帧。SFT 评估复用先前结果，已核对协议、模型资产和预测文件哈希。', '',
         '| 预测长度 | 指标 | 原始 SFT | 纯 Decisions RL | RL−SFT | RL 更好样本数 |',
         '|---|---|---:|---:|---:|---:|']
for h in ['4', '8']:
    for metric in ['lpips', 'mae', 'psnr', 'ssim']:
        row = paired[h][metric]
        lines.append(f"| {h} 帧 | {metric} {'↑' if metric in ('psnr','ssim') else '↓'} | {row['sft']:.6f} | {row['decisions']:.6f} | {row['decisions_minus_sft']:+.6f} | {row['decisions_wins']}/8 |")
lines += ['', 'LPIPS 差值的配对 bootstrap 95% 区间（单种子、8 样本；未经多重比较校正）：']
for h in ['4', '8']:
    lines.append(f"- {h} 帧：{paired[h]['lpips']['paired_bootstrap_95']}")
lines += ['', '| 8 帧运动诊断 | SFT | 纯 Decisions RL |', '|---|---:|---:|']
for metric in ['motion_roi_mae', 'predicted_motion', 'true_motion', 'delta_mae']:
    lines.append(f"| {metric} | {motion['sft'][metric]:.6f} | {motion['decisions'][metric]:.6f} |")
lines += ['', '运动区域按 GT 相对初始帧的平均 RGB 差 >0.02 定义，仅用于验证诊断。运动幅度不是越大越好。', '',
          '当前评分规则此前在 4 组静止负例中有 3 组偏好冻结帧；此轮按要求保留纯 Decisions 奖励进行检验。全图指标易受静态背景影响，不能替代物体运动/接触或机器人成功率。没有人工偏好标注。', '',
          f"API 新成功调用：{len(new_receipts)}；新输入 token：{report['new_input_tokens']}。已有官方评分仅在完整输入内容哈希一致时复用。", '',
          f"训练状态：{status['state']}。{status.get('error','')}", '',
          '实际 API 限流证据：`' + json.dumps(status['network']['errors'], ensure_ascii=False) + '`', '',
          f'Checkpoint：`{checkpoint}`', '',
          '仅为小规模探索。是否优于 SFT 以验证结果为准，不代表长期训练效果或任务成功率改善。']
(EXP / 'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report, indent=2))
