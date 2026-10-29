from pathlib import Path
import json,collections,re,os
D=Path(__file__).resolve().parent; ROOT=Path('/media/tyshao'); rows=json.loads((D/'inventory.json').read_text()); selected={}; exclusions=collections.Counter()
E='Jev-WM/experiments/'; S='fail_WM/FutureWorlds_SFT/'; R='fail_WM/RoboCasa/'
# Keep inference checkpoints that support reported comparisons, and resumable final milestones.
steps={
 'robocasa_object_memspo_20261024':[3],
 'robocasa_object_memspo_4gpu50_20261024':[15,20,40],
 'robocasa_object_memspo_4gpu_v2_20261024':[30],
 'robocasa_state_memspo_g8_8gpu_20261025':[15,30],
 'robocasa_state_memspo_g8_8gpu_s50_20261026':[50],
 'robocasa_state_v6_memspo_g8_8gpu_20261027':[15,30],
 'robocasa_traj_memspo_g8_8gpu_20261027':[15,30],
 'rt1_memspo200_upstream_20261017':[50,100,150,200],
 'rt1_qwen_a5_h32_20261020':[50,100], 'rt1_qwen_fm_h32_20261020':[50,100],
 'rt1_openai_a5_h32_20261021':[5,10],
 'rt1_openai_a5_h32_8gpu100_20261023':[10,14],
 'rt1_openai_a5_h32_8gpu256_20261023':[14],
}
full_rollouts={'robocasa_state_memspo_g8_8gpu_20261025','robocasa_state_v6_memspo_g8_8gpu_20261027','robocasa_traj_memspo_g8_8gpu_20261027'}
ck_roots={}
for n,ss in steps.items():
 for st in ss:
  p=ROOT/E/n/'train/checkpoints'/f'step_{st:06d}'
  if p.exists():ck_roots[str(p.relative_to(ROOT))]=(st==max(ss))
# All completed SFT ablations: final checkpoints; mixed codec: used, selected, final.
for p in (ROOT/S/'runs').glob('*/checkpoints'):
 dirs=sorted(x for x in p.glob('step_*') if x.is_dir() and not x.name.endswith('.partial'))
 if not dirs:continue
 chosen=[dirs[-1]]
 if p.parent.name=='paper_adapt_mixed_codec_20261009':chosen=[p/f'step_{i:06d}' for i in [34000,38000,50000]]
 for q in chosen:
  if q.exists():ck_roots[str(q.relative_to(ROOT))]=True
# Earlier local-success lineage: retain its final train checkpoint, rather than every smoke checkpoint.
for p in (ROOT/'Jev-WM/FutureWorlds/runs').glob('*/train/checkpoints'):
 dirs=sorted(x for x in p.glob('step_*') if x.is_dir() and not x.name.endswith('.partial'))
 if dirs:ck_roots[str(dirs[-1].relative_to(ROOT))]=True
text_ext={'.py','.sh','.json','.jsonl','.md','.txt','.toml','.yaml','.yml','.csv','.tsv','.html','.js','.css','.xml','.log','.ini','.cfg','.rst'}
media_ext={'.mp4','.png','.jpg','.jpeg','.webp','.gif','.svg','.pdf'}
weight_ext={'.pt','.pth','.safetensors','.bin','.ckpt','.model'}
root_assets=['Jev-WM/paper_assets/futureworlds_rt1/','Jev-WM/paper_assets/rlvr_world/',
 'Jev-WM/paper_assets/qwen/Qwen3-VL-8B-Instruct/',
 'fail_WM/FutureWorlds_workspace/assets/t5-base/','fail_WM/FutureWorlds_workspace/assets/rt1-upstream/',
 S+'assets/',S+'previews_wm_codec34k_20261012/natural_bundle/',S+'previews_wm_codec34k_20261012/success_bundle/',
 'Jev-WM/FutureWorlds/assets/']
for r in rows:
 p=Path(r['path']);rel=str(p.relative_to(ROOT));parts=p.parts;ext=p.suffix.lower();name=p.name;group=None;reason=None
 if any(x.startswith('.venv') or x in {'.git','.cache','__pycache__','node_modules','wandb'} for x in parts):reason='environment/cache'
 elif name in {'.env','.netrc','.git-credentials','hosts.yml'} or ext in {'.pem','.key','.pid','.lock','.pyc'} or name.endswith('.partial'):reason='runtime/credential/partial'
 elif 'cache2ms_backup' in rel or '/incomplete/' in rel or '/sync_incomplete/' in rel:reason='obsolete unsynchronized data or incomplete artifact'
 elif any(rel.startswith(x) for x in root_assets):group='model_dependencies'
 elif '/checkpoints/' in rel:
  match=next((k for k in ck_roots if rel.startswith(k+'/')),None)
  if match and (name!='training_state.pt' or ck_roots[match]):group='checkpoints'
  elif ext in text_ext:group='records'
  else:reason='unselected intermediate checkpoint/optimizer'
 elif rel.startswith(S+'robocasa_data/') or rel.startswith(E+'sibling_action_eval_20261025/data_extra/'):
  group='robocasa_packed_data'
 elif rel.startswith(R+'data/batch1000/') or rel.startswith(R+'data/batch1000_failure70/'):
  group='robocasa_raw_data'
 elif rel.startswith(R+'vendor/') or rel.startswith(R+'assets/'):
  group='simulator_runtime'
 elif rel.startswith(E+'rt1_memspo_20261017/data/') or rel.startswith('Jev-WM/data/'):
  group='earlier_task_data'
 elif '/judge_log/' in rel and ext=='.npz':
  exp=rel[len(E):].split('/')[0] if rel.startswith(E) else ''
  cur=name.split('_')[0]
  if exp in full_rollouts:group='rollouts_'+('state_v0' if exp=='robocasa_state_memspo_g8_8gpu_20261025' else 'state_v6' if 'state_v6' in exp else 'trajectory')
  elif cur.isdigit() and int(cur)<32:group='earlier_rollout_examples'
  else:reason='older full rollout videos; scores retained, first 32 cursors retained'
 elif ext in text_ext or name in {'LICENSE','Dockerfile','SOURCE_COMMIT','MANIFEST.in','.gitignore','.gitattributes'} or name.startswith('LICENSE'):
  group='code' if ext in {'.py','.sh','.toml'} or '/src/' in rel or '/source/' in rel or '/vendor/' in rel else 'records'
 elif ext in media_ext:group='review_media'
 elif ext in {'.npz','.npy','.pkl'}:
  if any(x in rel for x in ['/evaluation/','/evaluations/','/results/','/generated/','/frame_metrics/','/frames/','/traj_reward_20261027/','/calibration/','/cases/','/requests/','/reviews/']):group='evaluation_arrays'
  elif '/cache/' in rel or 'token_cache' in rel:reason='regenerable token cache'
  else:reason='unselected diagnostic intermediate'
 elif ext in weight_ext:reason='redundant exported bundle or unselected weights'
 else:reason='unselected binary/archive/runtime file'
 if group:selected[rel]={**r,'relative_path':rel,'group':group}
 else:exclusions[reason]+=r['bytes']
# External metric weights are needed for fully offline LPIPS evaluation.
p=Path('/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth')
if p.exists():selected['_runtime/torch/hub/checkpoints/'+p.name]=dict(path=str(p),relative_path='_runtime/torch/hub/checkpoints/'+p.name,bytes=p.stat().st_size,mtime=p.stat().st_mtime,symlink=None,group='metric_dependencies')
(D/'selected_files.json').write_text(json.dumps(list(selected.values()),indent=2))
(D/'checkpoint_selection.json').write_text(json.dumps(ck_roots,indent=2))
counts=collections.Counter();sizes=collections.Counter()
for r in selected.values():counts[r['group']]+=1;sizes[r['group']]+=r['bytes']
summary=dict(files=len(selected),logical_gib=sum(sizes.values())/2**30,groups={k:dict(files=counts[k],logical_gib=round(v/2**30,3)) for k,v in sizes.items()},excluded_gib={k:round(v/2**30,3) for k,v in exclusions.items()})
(D/'selection_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
