#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT=ROOT/'evals_codec34k_fullval_20261011'
r=json.loads((OUT/'report.json').read_text())
assert r['state']=='COMPLETE'
new=r['codecs']['mixed34k'];old=r['codecs']['old_success44k']
lines=['# 已部署混合 codec34k 独立复核','',
       '本次使用两组已训练 WM 配置中指定的实际 bundle，通过 WorldPipeline.encode_training_window 编码、CodecWorker.detokenize 两帧一组解码。重建输入是真实 RGB，不运行 WM 生成。',
       '全部100条验证轨迹（53成功、47失败），逐帧评估第2帧至末帧；FP32，matmul/cudnn TF32关闭，benchmark关闭。新旧codec使用同一数据及路径，最终test未使用。', '',
       '## 身份与复现', '',f'- 当前部署codec SHA256：`{new["codec_sha256"]}`。已与原step034000 checkpoint及两组WM run配置交叉检查。',
       f'- 新版51个冻结事件案例中，{new["previous_audit_exact_tracks"]}/51 的逐帧方块坐标与此前审计完全一致。',
       '- 这是相同验证集的独立实现复核和全帧扩展，不是新独立测试集上的泛化证明。', '',
       '## 全轨迹质量', '',
       '| 数据组 | codec | 帧数 | PSNR dB | 方块ROI误差 | 方块中心平均误差 px | 中心误差P95 px | GT可见但重建漏检 |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for group,title in [('success','成功'),('failure','失败')]:
    for label,c in [('旧success44k',old),('新mixed34k',new)]:
        s=c['summary'][group]
        lines.append(f"| {title} | {label} | {s['frames']} | {s['psnr_from_episode_mean_mse']:.2f} | {s['roi_mae_255']:.2f} | {s['center_error_px']:.2f} | {s['center_error_p95_px']:.2f} | {s['missed_frames']}/{s['gt_visible_frames']} ({s['miss_rate']:.2%}) |")
lines += ['', 'ROI以GT红色方块包围框向外扩8px定义，ROI和PSNR以episode等权汇总；中心误差按双方均检出的帧汇总，漏检以GT可见帧为分母。PSNR高不代表小物体事件正确，必须结合下面事件检查。此处为全轨迹/生成式解码指标，与训练日志的稀疏窗口/不同解码分块指标不可直接混算。', '',
          '## 冻结事件检查', '', '| Codec | 未抬起失败虚构抬升 /24 | 未抬起组窗口漏检帧 | 成功对照保留抬升 /20 | 提前释放落下 /7 | 提前释放漏检 /7 |',
          '|---|---:|---:|---:|---:|---:|']
for label,c in [('旧success44k',old),('新mixed34k',new)]:
    e=c['event_summary'];u=e['unlifted_failure'];s=e['success_control'];f=e['release_failure']
    lines.append(f"| {label} | {u['lifted']} | {u['missing_frames']}/{u['gt_visible_frames']} | {s['lifted']} | {f['fell_like_gt']} | {f['vanished']} |")
lines += ['',f"冻结抬升阈值{r['rise_threshold_px']:.6f}px，取事件窗口最大上移量，不能解释为首帧错误率。漏检表示红色方块检测器未检出，不直接等同物理消失。", '',
          '## 新codec残余异常', '']
for x in new['episodes']:
    e=x.get('event',{})
    if e.get('category')=='unlifted_failure' and (e['pred_lifted'] or e['vanished_frames']):
        lines.append(f"- `{x['id']}`：窗口最大上移{e['pred_rise_px']:.2f}px，虚构抬升={e['pred_lifted']}，窗口漏检{e['vanished_frames']}帧。")
lines += ['', '### 虚构抬升阈值敏感性', '', '| codec | >18px | 冻结阈值 | >25px | >30px | >40px |','|---|---:|---:|---:|---:|---:|']
for label,c in [('旧success44k',old),('新mixed34k',new)]:
    xs=[x['event']['pred_rise_px'] for x in c['episodes'] if x.get('event',{}).get('category')=='unlifted_failure']
    lines.append('| '+label+' | '+' | '.join(str(sum(v is not None and v>t for v in xs)) for t in [18,r['rise_threshold_px'],25,30,40])+' |')
lines += ['', '## 配对episode的改善/退步', '', '| 数据组 | 新版ROI较低 | 较高 | 相等 |', '|---|---:|---:|---:|']
old_by_id={x['id']:x for x in old['episodes']}
for flag,name in [(True,'成功'),(False,'失败')]:
    ds=[x['roi_mae_255']-old_by_id[x['id']]['roi_mae_255'] for x in new['episodes'] if x['success']==flag and x['roi_mae_255'] is not None and old_by_id[x['id']]['roi_mae_255'] is not None]
    lines.append(f'| {name} | {sum(d<0 for d in ds)} | {sum(d>0 for d in ds)} | {sum(d==0 for d in ds)} |')
e=new['event_summary'];passed=e['unlifted_failure']['lifted']<=1 and e['unlifted_failure']['undetected']==0 and e['success_control']['lifted']==20 and e['release_failure']['fell_like_gt']==7
lines += ['', '## 结论与使用范围','',
          f'预先采用的事件验收条件：{"通过" if passed else "未通过"}。该验收允许最多1/24虚构抬升，因此通过不能表述为完全没有codec乐观偏差。',
          '用于新WM对照是否合理，应综合查看新旧质量表、事件表及人工核查；任何残余codec错误应随WM评估单列。此次没有更换已训练WM的codec，也没有重新训练。', '',
          '7条案例分别保存GT/重建MP4：正常成功、抓取失败、提前释放、旧版难例及新版本残余异常。文件前缀mixed34k/old_success44k标明codec版本，左GT右重建。全部留在服务器。',
          '逐帧原始检测坐标、距离、漏检帧号及事件结果：`report.json`；关键帧为`*_audit.png`。']
(OUT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
