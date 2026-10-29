import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'rt1_memspo200_upstream_20261017'
def read(p):return json.loads(Path(p).read_text())


def main():
    steps=[json.loads(x) for x in (ROOT/'train/metrics.jsonl').read_text().splitlines()]
    # A restored checkpoint can cause replayed log rows; use the last committed
    # occurrence of each scheduled batch, then verify a complete 1..200 record.
    steps={r['step']:r for r in steps};assert sorted(steps)==list(range(1,201))
    totals={k:sum(r['judge'].get(k,0) for r in steps.values()) for k in ['accepted_pairs','judged_pairs','order_conflicts','tied_answers','insufficient_answers']}
    updates=sum(r['optimizer_updated'] for r in steps.values())
    result=dict(protocol=read(ROOT/'protocol.json'),scheduled_batches=200,optimizer_updates=updates,judge=totals,splits={})
    lines=['# RT-1：SFT / R0 MemSPO200 / 纯 Jev-GT200','',
           f'4卡，global batch16，训练预测10帧。200个计划批次，实际optimizer更新{updates}次。无R0奖励或R0候选筛选。',
           f'有效候选对：{totals["accepted_pairs"]:.0f}/{totals["judged_pairs"]:.0f}；同顺序标签冲突：{totals["order_conflicts"]:.0f}。',
           'Jev 只在训练奖励端看到未来GT；生成和评测不注入GT。固定最终batch200，不按测试结果选checkpoint。','',
           '|划分|预测帧数|指标|SFT|原R0 RL|纯Jev RL|Jev相对SFT改善|Jev相对R0改善|',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for split in ['validation','test']:
        runs={name:read(path) for name,path in [('sft',ROOT/f'evaluation/{split}/sft/summary.json'),
                ('r0',BASE/f'evaluation/{split}/rl/summary.json'),('jev',ROOT/f'evaluation/{split}/rl/summary.json')]}
        ids=[x['id'] for x in runs['sft']['per_case']]
        for r in runs.values():
            assert [x['id'] for x in r['per_case']]==ids
            for k in ['manifest_sha256','horizons','beam_width','memory','precision','tf32','initial_observations','gt_refresh','text_assets']:
                assert r['protocol'][k]==runs['sft']['protocol'][k],k
        # Only training.py and newly added judge modules differ from baseline;
        # inference and metric source files must remain byte-identical.
        upstream=read(ROOT/'upstream_source_hashes.json')
        import hashlib
        for name,h in upstream.items():
            if name!='src/futureworlds/training.py':
                assert hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()==h
        split_result={}
        for h in ['10','20','32']:
            values={}
            for metric in ['lpips','psnr','ssim','mae','mse']:
                arrays={name:np.array([x['metrics'][h][metric] for x in r['per_case']]) for name,r in runs.items()}
                means={name:float(a.mean()) for name,a in arrays.items()};comparisons={}
                sign=-1 if metric in ['lpips','mae','mse'] else 1
                for baseline in ['sft','r0']:
                    delta=arrays['jev']-arrays[baseline]
                    rng=np.random.default_rng(20261017)
                    ci=np.quantile(delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1),[.025,.975])
                    comparisons[baseline]=dict(delta=float(delta.mean()),relative_improvement=100*sign*float(delta.mean())/means[baseline],
                                               paired_episode_bootstrap_95ci=ci.tolist(),improved_cases=int((sign*delta>0).sum()))
                values[metric]=dict(means=means,comparisons=comparisons,cases=len(ids))
                # PSNR is logarithmic: display dB difference, not percent gain.
                def fmt(b):
                    c=comparisons[b]
                    return f'{c["delta"]:+.3f} dB' if metric=='psnr' else f'{c["relative_improvement"]:+.2f}%'
                lines.append(f'|{split}|{h}|{metric}|{means["sft"]:.6f}|{means["r0"]:.6f}|{means["jev"]:.6f}|{fmt("sft")}|{fmt("r0")}|')
            split_result[h]=values
        result['splits'][split]=split_result
    lines+=['','一次训练种子、本地数据划分；测试集此前已用于查看R0结果，因此是探索性比较。图像指标不等于任务成功率。',
            '评分覆盖率和实际更新次数必须与指标一起解释；不能把所有涨跌归因为裁判语义正确性。']
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
