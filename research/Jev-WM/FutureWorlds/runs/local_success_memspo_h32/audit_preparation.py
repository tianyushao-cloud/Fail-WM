import json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'scripts')
from scholar_success_memspo import verify_source
from local_success_memspo import assert_episode_start
from futureworlds.checkpoints import file_sha256
r=Path('/media/tyshao/Jev-WM/FutureWorlds')
old=r/'runs/local_success_memspo';new=r/'runs/local_success_memspo_h32'
verify_source();assert_episode_start(SimpleNamespace(output=str(new),gpus=4,horizon=32))
a=json.loads((old/'train.json').read_text());b=json.loads((new/'train.json').read_text())
paths={'output','train_manifest','evaluation_manifest','schedule'}
diff={k:dict(old=a.get(k),new=b.get(k)) for k in set(a)|set(b) if a.get(k)!=b.get(k) and k not in paths}
assert diff=={'horizon':{'old':10,'new':32}},diff
assert json.loads((old/'schedule.json').read_text())==json.loads((new/'schedule.json').read_text())
audit={'hyperparameter_changes':diff,'identical_schedule':True,'source_lock_pass':True,'splits':{}}
for split in ['train','validation','test']:
 m=new/f'data/{split}/manifest.json';previous=json.loads((old/f'data/{split}/manifest.json').read_text());present=json.loads(m.read_text())
 assert previous==present
 for row in present['cases']:
  assert row['source_start']==0 and row['success'] is True and row['frames']>=34
  assert file_sha256(m.parent/row['file'])==row['sha256']
 audit['splits'][split]=dict(cases=len(present['cases']),same_inputs_as_h10=True,all_npz_hashes_verified=True,all_source_start_zero=True)
# A mistaken horizon10 output must be rejected before any stage can resume it.
try:assert_episode_start(SimpleNamespace(output=str(old),gpus=4,horizon=32))
except ValueError: audit['cross_horizon_resume_rejected']=True
else:raise AssertionError('Cross-horizon run was accepted')
audit['formal_run_started']=(new/'train/run.json').exists()
(new/'preparation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))

