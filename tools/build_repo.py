from pathlib import Path
import json,shutil,hashlib,subprocess,os,collections
D=Path(__file__).resolve().parent;repo=D/'repo';repo.mkdir(exist_ok=True);rs=json.loads((D/'selected_files.json').read_text());copied=[]
latest=Path('/media/tyshao/Jev-WM/experiments/robocasa_traj_memspo_g8_8gpu_20261027/source')
for name in ['src','tests','licenses']:
 shutil.copytree(latest/name,repo/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['pyproject.toml','THIRD_PARTY_NOTICES.md']:
 shutil.copy2(latest/name,repo/name)
important={'train.json','run.json','summary.json','complete.json','comparison.json','protocol.json','sampling.json','schedule.json','launch_identity.json','launch.json','train_launch.json','source_hashes.json','upstream_source_hashes.json','preparation_check.json','selection.json','codec_selection.json','protocol_used.json','case_table.json','run_8gpu.sh','report.txt','report2.txt','report3.txt','fidelity.txt','REPORT.md'}
for r in rs:
 p=Path(r['path']);rel=r['relative_path']
 keep=r['group']=='code' or (r['group']=='records' and (p.name in important or p.suffix in {'.md','.txt','.csv'} or any(x in p.parts for x in ['configs','provenance','licenses'])))
 if not keep or r['bytes']>8*2**20:continue
 target=repo/'research'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 copied.append(dict(path=str(target.relative_to(repo)),source=r['path'],bytes=r['bytes']))
(repo/'migration').mkdir(exist_ok=True)
for name in ['selection_summary.json','checkpoint_selection.json','directory_links.json','secret_scan_findings.json']:
 shutil.copy2(D/name,repo/'migration'/name)
(repo/'migration/source_map.json').write_text(json.dumps(copied,indent=2))
(repo/'.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\n.env\n.env.*\n*.key\n*.pem\nrelease_assets/\nrestored/\n*.tar.zst.part*\n')
# Capture actual environment package versions, without pip URLs/credential-bearing settings.
py='/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python'
script='import importlib.metadata as m,json,platform; print(json.dumps(dict(python=platform.python_version(),packages={d.metadata["Name"]:d.version for d in m.distributions()}),indent=2))'
x=subprocess.run([py,'-c',script],check=True,capture_output=True,text=True)
(repo/'migration/environment_inventory.json').write_text(x.stdout)
run=json.loads((latest.parent/'train/run.json').read_text())
(repo/'migration/training_environment.json').write_text(json.dumps(run['config']['implementation'],indent=2))
(repo/'requirements-reproduction.txt').write_text('# Core versions are recorded by the completed RL run; select a compatible CUDA wheel index for torch.\n'+'\n'.join(f'{k}=={v}' for k,v in run['config']['implementation']['versions'].items())+'\n'+ '\n'.join(k for k in ['h5py','scipy','opencv-python-headless','einops','safetensors','Pillow','protobuf','sentencepiece','huggingface_hub','imageio','imageio-ffmpeg'])+'\n')
print('Prepared repository source/config/history files',len(copied))
