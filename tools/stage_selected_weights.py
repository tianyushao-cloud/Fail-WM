from pathlib import Path
import json,os,hashlib,shutil,collections
D=Path(__file__).resolve().parent;out=D/'hf_model';out.mkdir(exist_ok=True);root=Path('/media/tyshao');E=root/'Jev-WM/experiments';S=root/'fail_WM/FutureWorlds_SFT';W=root/'fail_WM/FutureWorlds_workspace'
items=[
 ('bundles/robocasa_original_release',S/'assets/futureworlds_scholar_robocasa','上游原始 RoboCasa 发布 bundle，原 manifest 保持完整'),
 ('bundles/rt1_original_release',root/'Jev-WM/paper_assets/futureworlds_rt1','早期 RT-1 对照的原始发布 bundle'),
 ('initialization/rt1_upstream',W/'assets/rt1-upstream','混合数据 SFT/codec 的实际原始初始化'),
 ('bundles/robocasa_rt1_initialization',S/'assets/futureworlds_bundle_rt1_init','原始 RT-1 初始化适配到 RoboCasa 的配置与动作范围'),
 ('bundles/robocasa_mixed_codec34k_initialization',S/'assets/futureworlds_bundle_mixed_codec_step034000_20261011','实际训练混合 SFT10k 的 codec34k 初始化 bundle'),
 ('bundles/robocasa_mixed_sft10k',S/'previews_wm_codec34k_20261012/natural_bundle','全部主要 RoboCasa RL 的 mixed SFT10k 初始化，可直接推理'),
 ('bundles/robocasa_success_sft10k',S/'previews_wm_codec34k_20261012/success_bundle','成功-only SFT10k 对照，可直接推理'),
 ('codecs/robocasa_mixed_step034000',S/'runs/paper_adapt_mixed_codec_20261009/checkpoints/step_034000','实际配套 codec34k；含优化器/判别器/RNG，支持续训'),
 ('checkpoints/robocasa_sft_mixed_step010000',S/'runs/natural_sft_mixed_codec34000_10k_20261011/checkpoints/step_010000','混合 SFT10k 完整 checkpoint'),
 ('checkpoints/robocasa_sft_success_step010000',S/'runs/success_only_sft_mixed_codec34000_10k_20261011/checkpoints/step_010000','成功-only SFT10k 完整 checkpoint'),
 ('checkpoints/robocasa_rl_state_v0_step000030',E/'robocasa_state_memspo_g8_8gpu_20261025/train/checkpoints/step_000030','已做 val/test 的 V0 状态奖励 RL30 对照'),
 ('checkpoints/robocasa_rl_state_v6_step000030',E/'robocasa_state_v6_memspo_g8_8gpu_20261027/train/checkpoints/step_000030','最新 V6 状态奖励 RL30；未找到正式效果评测'),
 ('checkpoints/robocasa_rl_trajectory_step000030',E/'robocasa_traj_memspo_g8_8gpu_20261027/train/checkpoints/step_000030','最新纯二维轨迹奖励 RL30；未找到正式效果评测'),
 ('dependencies/t5-base',W/'assets/t5-base','推理必需、体积小，固定原始 T5 版本，支持离线加载'),
]
known={r['path']:r for r in json.loads((D/'hashed_files.json').read_text())};files=[];models=[]
for dest,source,reason in items:
 if not source.is_dir():raise FileNotFoundError(source)
 total=0;n=0
 for p in sorted(source.rglob('*')):
  if not p.is_file() or '.cache' in p.parts:continue
  target=out/dest/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
  r=known.get(str(p))
  if r:h=r['sha256']
  else:
   hh=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*2**20),b''):hh.update(b)
   h=hh.hexdigest()
  if not target.exists():os.link(p,target)
  files.append(dict(path=str(target.relative_to(out)),bytes=p.stat().st_size,sha256=h,original_path=str(p),restore_path=str(p.relative_to(root))))
  total+=p.stat().st_size;n+=1
 models.append(dict(path=dest,original_path=str(source),reason=reason,bytes=total,files=n))
manifest=dict(schema=1,github='tianyushao-cloud/Fail-WM',repo_id='tianyushao1124/Fail-WM',private=True,scope='Selected original initialization, codec34k, mixed/success SFT10k, V0/V6/trajectory RL30; no training data or bulk historical rollouts',models=models,files=files)
(out/'weights_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(D/'repo/migration/weights_manifest.json').write_bytes((out/'weights_manifest.json').read_bytes())
summary=dict(files=len(files),logical_gib=sum(x['bytes'] for x in files)/2**30,unique_gib=sum({x['sha256']:x['bytes'] for x in files}.values())/2**30,models=models)
(D/'repo/migration/selected_weights_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('Selected files',len(files),'logical GiB',round(summary['logical_gib'],3),'unique GiB',round(summary['unique_gib'],3))
for x in models:print(x['path'],round(x['bytes']/2**30,3))
