import json,time,os,subprocess,shutil,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent;data=root/'data/batch1000';repair_pid=int((root/'sampling_fix.pid').read_text())
while not (data/'sampling_fix_report.json').exists():
    if (root/'CANCEL_BATCH1000').exists():raise SystemExit('cancel requested')
    try:os.kill(repair_pid,0)
    except ProcessLookupError:raise SystemExit('sampling repair ended without a complete report')
    time.sleep(5)
report=json.loads((data/'sampling_fix_report.json').read_text());assert len(report)==574,len(report)
rows=[json.loads(p.read_text()) for p in (data/'episodes').glob('*/metadata.json')];assert len(rows)==569,len(rows)
assert all(m['complete'] and m['validation']['sampling_sync_pass'] and m['validation']['replay_pass'] for m in rows)
old=data/'source_code_policy_v2_0'
if not old.exists():shutil.copytree(root/'data/batch1000_cache2ms_backup/source_code',old)
for name in ['BATCH1000_README.md','batch_report.py','audit_action_conversion.py','verify.py']:shutil.copy2(root/name,data/'source_code'/name)
provenance=json.loads((root/'provenance.json').read_text());provenance.update(observation_schema_version='2.1',action_policy_versions=['2.0_cached_feedback (native actions preserved and re-recorded)','2.1_synchronized_feedback'],native_action_dimension=12,workers=4,
    source_files_sha256={str(p.relative_to(data/'source_code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (data/'source_code').rglob('*') if p.is_file()})
(data/'environment_provenance.json').write_text(json.dumps(provenance,indent=2))
log=(root/'batch1000_after_sync.log').open('ab',buffering=0)
p=subprocess.Popen([str(root/'.venv/bin/python'),str(root/'batch_collect.py'),'--output','data/batch1000','--episodes','1000','--workers','4'],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'batch1000.pid').write_text(str(p.pid));print('RESUMED',p.pid,'validated existing',len(rows),flush=True)
