"""Summarize step-one probes without turning sensitivity into causal attribution."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT=ROOT/'diagnostics_step1_20261012'
def load(p):return json.loads(Path(p).read_text())
def summ(values):
    x=np.asarray(values,float)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)))
def main():
    protocol=load(OUT/'protocol.json');done=load(OUT/'complete.json');assert done['state']=='COMPLETE'
    frames=[load(OUT/'frames'/f['key']/'result.json') for f in protocol['frames']]
    histories=[load(p) for p in sorted((OUT/'history').glob('*/result.json'))]
    assert len(frames)==20 and len(histories)==10
    local=[v for f in frames for v in f['locality'] if not v.get('zero_effect')]
    probes=[v for f in frames for v in f['probe_records'] if not v.get('zero_effect')]
    locality={k:summ([v[k] for v in local]) for k in ('area50_fraction','area90_fraction','nominal_tile_energy','neighbor3x3_energy','peak_in_nominal_tile')}
    locality['per_intervention_kind']={str(j):{k:summ([v[k] for v in probes if v['kind']==j]) for k in ('area50_fraction','area90_fraction','nominal_tile_energy','neighbor3x3_energy','peak_in_nominal_tile')} for j in range(3)}
    active=[v for f in frames for v in sorted(f['locality'],key=lambda x:x['total'],reverse=True)[:20]]
    locality['strongest20_tokens_per_frame']={k:summ([v[k] for v in active]) for k in ('area50_fraction','area90_fraction','nominal_tile_energy','neighbor3x3_energy')}
    locality['roi_influence_n50']=summ([f['roi_influence_n50'] for f in frames]);locality['roi_influence_n90']=summ([f['roi_influence_n90'] for f in frames])
    locality['nominal_roi_tokens']=summ([f['nominal_roi']['tokens'] for f in frames])
    locality['nominal_roi_influence_share']=summ([f['nominal_roi_influence_share'] for f in frames])
    locality['batch_max_error']=max(f['batch_vs_serial_max_pixel_error'] for f in frames)
    locality['paired_other_frame_max_change']=max(f['other_frame_max_pixel_change'] for f in frames)
    locality['total_fallback_positions']=sum(len(f['all_train_donor_fallback_positions']) for f in frames)
    # Different replacements must agree for a fixed, robust spatial mask.
    locality['donor1_vs_donor2_roi_energy_pearson']=summ([f['replacement_kind_roi_rank_correlation'][0][1] for f in frames])
    locality['donor1_vs_nearest_roi_energy_pearson']=summ([f['replacement_kind_roi_rank_correlation'][0][2] for f in frames])
    region={}
    for role in ('all','success','failure'):
        ff=[f for f in frames if role=='all' or f['frame']['role']==role]
        region[role]={'frames':len(ff)}
        for partition in ('nominal_roi','empirical_roi90'):
            region[role][partition]={k:summ([f[partition][k] for f in ff]) for k in ('tokens','mean_nll','outside_mean_nll','loss_share')}
        for metric in ('codec_rgb','wm_beam_nextframe_rgb','wm_vs_codec_rgb'):
            region[role][metric]={k:summ([f[metric][k] for f in ff]) for k in ('roi_mae255','outside_mae255','full_mae255')}
    directions=[d for h in histories for d in h['directions']]
    history_by_factor={}
    for factor in sorted({h['pair']['factor'] for h in histories}):
        ds=[d for h in histories if h['pair']['factor']==factor for d in h['directions']]
        history_by_factor[factor]=dict(directions=len(ds),
            own_target_prefers_own_history=sum(d['conditional_scores'][0]['nll_swapped_minus_own']>0 for d in ds),
            donor_target_prefers_donor_history=sum(d['conditional_scores'][1]['nll_swapped_minus_own']<0 for d in ds),
            free_beam_token_changed_fraction=summ([d['free_beam_token_changed_fraction'] for d in ds]),
            first_token_js=summ([d['conditional_scores'][0]['visual_js_first_token'] for d in ds]))
    history={k:summ([d[k] for d in directions]) for k in ('history_visual_token_changed_fraction','free_beam_token_changed_fraction')}
    history['image_change']={k:summ([d['free_beam_image_change'][k] for d in directions]) for k in ('roi_mae255','outside_mae255','full_mae255')}
    for role in ('all','success','failure'):
        dd=[d for d in directions if role=='all' or (d['base']==next(h['pair'][role] for h in histories if d in h['directions']))]
        own=[d['conditional_scores'][0] for d in dd]
        # For donor target, matching history is the swapped history.
        donor=[d['conditional_scores'][1] for d in dd]
        history[role]=dict(directions=len(dd),own_target_prefers_own_history=sum(x['nll_swapped_minus_own']>0 for x in own),
                          donor_target_prefers_donor_history=sum(x['nll_swapped_minus_own']<0 for x in donor),
                          own_target_delta_nll=summ([x['nll_swapped_minus_own'] for x in own]),
                          donor_target_delta_nll=summ([x['nll_swapped_minus_own'] for x in donor]),
                          first_token_js=summ([x['visual_js_first_token'] for x in own]),
                          mean_conditional_js=summ([x['visual_js_mean'] for x in own]),
                          roi_union_nll_delta=summ([x['roi_union_nll_swapped_minus_own'] for x in own]))
    # Export influence and NLL grids aligned with GT for human review.
    for f in frames:
        path=OUT/'frames'/f['frame']['key']
        with np.load(path/'arrays.npz') as a:roi_imp=a['roi_influence'];nll=a['nll']
        cv=Image.new('RGB',(960,304),'white');dr=ImageDraw.Draw(cv)
        pic=Image.open(path/'gt_codec_nextframe.jpg').crop((0,24,320,280));cv.paste(pic,(0,24))
        dr.text((5,5),f['frame']['key']+' GT',fill='black')
        for offset,vals,label in [(320,roi_imp,'Pixel influence inside cube ROI'),(640,nll,'A teacher-forced NLL per token')]:
            norm=vals/(vals.max()+1e-12)
            for i,v in enumerate(norm):
                y,x=divmod(i,10);col=(int(255*v),int(210*(1-v)),int(230*(1-v)))
                dr.rectangle((offset+x*32,24+y*32,offset+x*32+31,24+y*32+31),fill=col,outline='white')
                dr.text((offset+x*32+2,24+y*32+2),str(i),fill='black')
            dr.text((offset+4,5),label,fill='black')
        dr.text((4,284),'Influence is intervention amplitude dependent; token grid is nominal 8x10.',fill='black');cv.save(path/'influence_nll.jpg')
    result=dict(state='COMPLETE',completion=done,locality=locality,regional_error=region,history=history,history_by_factor=history_by_factor,
                per_frame=[dict(key=f['frame']['key'],role=f['frame']['role'],factor=f['frame']['factor'],nominal=f['nominal_roi'],empirical90=f['empirical_roi90']) for f in frames],
                limitations=protocol['limitations']+['History exchanges entire recent visual state, including arm pose, and may produce off-manifold combinations. Sensitivity is not proof of correct cube-state use.',
                    'Repeated true-target tokens inside each next-frame teacher-forced forward can help later tokens. First-token JS and free beam images are reported separately.',
                    'Perturbation energy depends on alternative code. Top90 influence tokens are an exploratory empirical proxy, not yet a validated loss mask.',
                    'Pixel errors are RGB MAE and not perceptual quality or task success. No optimization or generalization claim.',
                    'Per-frame replacement_kind_roi_rank_correlation field contains Pearson correlations of absolute ROI energies, not rank/Spearman correlations; summary names correct this.'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def med(key):return locality[key]['median']
    lines=['# 第一步诊断结果','','完成：20个train帧、5个初态成功失败配对、4800次单token干预、20次固定动作历史交换。没有训练WM。','',
           '## 1. token与像素影响','','三种替换平均后，每个token影响能量覆盖范围（最强像素累计能量）：',
           f"- 50%能量面积中位数 {med('area50_fraction'):.2%}；90%能量面积中位数 {med('area90_fraction'):.2%}。",
           f"- 名义32×32格内能量中位数 {med('nominal_tile_energy'):.2%}；邻近3×3格内 {med('neighbor3x3_energy'):.2%}。",
           f"- 方块框外扩8px跨名义token数中位数 {med('nominal_roi_tokens'):.1f}；覆盖方块ROI扰动能量50%/90%所需token中位数 {med('roi_influence_n50'):.1f}/{med('roi_influence_n90'):.1f}。",
           f"- 名义ROI网格包含的方块影响能量中位数 {med('nominal_roi_influence_share'):.2%}。",
           f"- 仅看每帧影响最强的20个token：90%能量面积中位数 {locality['strongest20_tokens_per_frame']['area90_fraction']['median']:.2%}，邻近3×3格能量中位数 {locality['strongest20_tokens_per_frame']['neighbor3x3_energy']['median']:.2%}。局部性确实存在；全token统计包含弱背景变化的长尾，不能解释为每个token都同等改变全图。",
           f"- 批量/原路径最大像素差 {locality['batch_max_error']:.3g}；相邻未改帧最大差 {locality['paired_other_frame_max_change']:.3g}（RGB范围0..1）。",
           '', '## 2. 方块附近的预测误差','','|类别|N|名义ROI token数均值|ROI token NLL|其他token NLL|ROI loss占比|codec ROI MAE|WM ROI MAE|WM非ROI MAE|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for role in ('all','success','failure'):
        r=region[role];n=r['nominal_roi'];lines.append(f"|{role}|{r['frames']}|{n['tokens']['mean']:.2f}|{n['mean_nll']['mean']:.3f}|{n['outside_mean_nll']['mean']:.3f}|{n['loss_share']['mean']:.2%}|{r['codec_rgb']['roi_mae255']['mean']:.2f}|{r['wm_beam_nextframe_rgb']['roi_mae255']['mean']:.2f}|{r['wm_beam_nextframe_rgb']['outside_mae255']['mean']:.2f}|")
    lines+=['','NLL用训练实际完整词表CE；区域划分同时保存名义格与探针90%影响token两种版本，详见JSON。RGB用真实历史下自由生成下一帧beam4，数值范围0..255。','',
            '## 3. 相同动作，交换最近真实历史','',
            f"- 自由生成下一帧token变化率中位数 {history['free_beam_token_changed_fraction']['median']:.2%}。",
            f"- 下一帧RGB变化MAE中位数 {history['image_change']['full_mae255']['median']:.3f}。",
            f"- 自己的真实target偏好自己的历史：{history['all']['own_target_prefers_own_history']}/20；对方target偏好对方历史：{history['all']['donor_target_prefers_donor_history']}/20。",
            f"- 不使用当前帧任何GT token的首token JS均值 {history['all']['first_token_js']['mean']:.5f} nats；全80token条件JS均值 {history['all']['mean_conditional_js']['mean']:.5f}。",
            '- 固定动作/context/anchor断言通过；重复同一输入的NLL完全相同。',
            '', '历史交换也会改变机械臂姿态，不能仅据敏感性认定模型正确利用了方块。所有比较为5组训练样本诊断，不是新模型性能结果。',
            '', '## 产物','', '`summary.json`：完整数值；`frames/*/influence_nll.jpg`：GT/影响网格/NLL；`frames/*/gt_codec_nextframe.jpg`：GT/codec/WM下一帧；`history/*/*.jpg`：历史交换前后对照。']
    (OUT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
