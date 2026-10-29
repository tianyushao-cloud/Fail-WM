from pathlib import Path
import json,hashlib,collections,csv
D=Path(__file__).resolve().parent;repo=D/'repo';rows=json.loads((D/'hashed_files.json').read_text());byrel={r['relative_path']:r for r in rows};bysource={r['path']:r for r in rows};root=Path('/media/tyshao')
issues=[];ckrows=[];cks=json.loads((D/'checkpoint_selection.json').read_text())
def inspect_file(p,expected=None):
 p=p.resolve();r=bysource.get(str(p))
 if r is None:return 'not_in_archive' if p.exists() else 'missing_on_server'
 if expected and r['sha256']!=expected:return 'checksum_mismatch'
 return None
for rel in cks:
 p=root/rel;j=json.loads((p/'experiment.json').read_text());bad=[]
 for name,h in j['files'].items():
  error=inspect_file(p/name,h)
  if error:bad.append(dict(file=name,error=error))
 cfg=j.get('config',{});entry=dict(path=rel,step=j.get('step'),dataset=cfg.get('dataset'),method=cfg.get('method'),reward=cfg.get('reward'),model_sha256=j['files'].get('model.safetensors',j['files'].get('diffusion_pytorch_model.safetensors')),complete_with_optimizer=(p/'training_state.pt').as_posix() in bysource,errors=bad)
 ckrows.append(entry)
 if bad:issues.append(dict(checkpoint=rel,errors=bad))
refs=0;manifests=0;bundle_refs=0;bundle_missing=[]
for r in rows:
 p=Path(r['path'])
 if p.suffix!='.json' or r['bytes']>8*2**20:continue
 if not ('robocasa_data' in str(p) or 'data_extra' in str(p) or p.name=='manifest.json'):continue
 try:j=json.loads(p.read_text())
 except (ValueError,OSError):continue
 if not isinstance(j,dict):continue
 if isinstance(j.get('cases'),list):
  manifests+=1
  for row in j['cases']:
   if not isinstance(row.get('file'),str):continue
   tocheck=[(p.parent/row['file'],row.get('sha256'))]
   if row.get('source_h5'):tocheck.append((Path(row['source_h5']),row.get('source_h5_sha256')))
   for q,h in tocheck:
    refs+=1;error=inspect_file(q,h)
    if error:issues.append(dict(manifest=r['relative_path'],dependency=str(q),error=error))
 if isinstance(j.get('files'),list):
  for f in j['files']:
   if not isinstance(f,dict) or 'path' not in f or 'sha256' not in f:continue
   bundle_refs+=1;error=inspect_file(p.parent/f['path'],f['sha256'])
   if error:bundle_missing.append(dict(manifest=r['relative_path'],dependency=str(p.parent/f['path']),expected_sha256=f['sha256'],error=error))
source_checks=[]
for name in ['robocasa_traj_memspo_g8_8gpu_20261027','robocasa_state_v6_memspo_g8_8gpu_20261027','robocasa_state_memspo_g8_8gpu_20261025']:
 p=root/'Jev-WM/experiments'/name;j=json.loads((p/'launch_identity.json').read_text());bad=[]
 for rel,h in j['source'].items():
  error=inspect_file(p/rel,h)
  if error:bad.append(rel)
 source_checks.append(dict(experiment=name,source_files=len(j['source']),mismatches=bad))
 issues.extend(dict(source=x['experiment'],mismatches=x['mismatches']) for x in source_checks if x['mismatches'])
summary=dict(state='PASS' if not issues else 'REVIEW_REQUIRED',selected_files=len(rows),full_checkpoints=len(ckrows),dataset_manifests=manifests,dataset_file_references_checked=refs,bundle_file_references_checked=bundle_refs,issues=issues,exported_bundle_gaps=bundle_missing,launch_source_checks=source_checks)
(repo/'migration/dependency_audit.json').write_text(json.dumps(summary,indent=2));(repo/'migration/checkpoint_registry.json').write_text(json.dumps(ckrows,indent=2))
for name in ['dedup_summary.json','checkpoint_selection.json','live_status_exclusions.json','missing_dataset_dependencies.json']:
 (repo/'migration'/name).write_bytes((D/name).read_bytes())
counts=collections.Counter();sizes=collections.Counter();unique={}
for r in rows:counts[r['group']]+=1;sizes[r['group']]+=r['bytes'];unique.setdefault(r['sha256'],r)
s=dict(files=len(rows),unique_files=len(unique),logical_gib=sum(sizes.values())/2**30,unique_gib=sum(r['bytes'] for r in unique.values())/2**30,groups={k:dict(files=counts[k],logical_gib=round(v/2**30,3)) for k,v in sizes.items()},selection_note='Selected checkpoints include every file checked by their original receipts; exports deduplicated by content. Older intermediate checkpoints retain metadata only. No credentials or virtualenvs.')
(repo/'migration/selection_summary.json').write_text(json.dumps(s,indent=2));(D/'selection_summary.json').write_text(json.dumps(s,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['issues','exported_bundle_gaps']},indent=2));print('ISSUES',json.dumps(issues[:8]));print('BUNDLE GAPS',len(bundle_missing),json.dumps(bundle_missing[:8]))
