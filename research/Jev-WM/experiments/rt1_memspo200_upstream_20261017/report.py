import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

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
    text = '# RT-1 SFT → MemSPO200 对照\n\n从发布 SFT50k 初始化，200 步、global batch16、horizon10、R0，多卡DDP，实际卡数见launch.json。固定最终200步，无测试集选模。\n\n'
    text += f'测试集32帧 LPIPS：SFT {primary["sft"]:.6f} → RL {primary["rl"]:.6f}；相对改善 {primary["relative_improvement_percent"]:+.2f}%；配对 episode bootstrap 95% CI（RL−SFT）[{ci[0]:.6f}, {ci[1]:.6f}]。\n\n'
    text += '| 划分 | 帧数 | 指标 | SFT | RL | RL−SFT | 相对改善 |\n|---|---:|---|---:|---:|---:|---:|\n' + '\n'.join(table)
    text += '\n\n正的相对改善表示指标方向变好。区间跨零时不声称稳健提升。一次训练种子；测试 episode 与本次RL隔离，但原SFT训练集重合未知。图像预测指标不等于机器人成功率。\n'
    (ROOT / 'REPORT.md').write_text(text)
    return primary

if __name__ == '__main__':
    summarize()
