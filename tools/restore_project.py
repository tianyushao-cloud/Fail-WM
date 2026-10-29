#!/usr/bin/env python3
"""Verify selected weights and restore original relative code/model paths. No old-server access."""
import argparse,hashlib,json,shutil,os
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()

def safe(root,rel):
 p=Path(rel)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe relative path')
 result=root/p
 if not result.resolve().is_relative_to(root.resolve()):raise ValueError('Destination escapes selected root')
 return result

def install(src,dest,sha):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if dest.stat().st_size!=src.stat().st_size or digest(dest)!=sha:raise FileExistsError('Existing file differs: '+str(dest))
  return
 tmp=dest.with_name(dest.name+'.restore-partial')
 with src.open('rb') as fi,tmp.open('xb') as fo:shutil.copyfileobj(fi,fo,8*2**20)
 if digest(tmp)!=sha:raise RuntimeError('Restored file checksum mismatch: '+str(dest))
 shutil.copystat(src,tmp);tmp.replace(dest)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--destination',type=Path);ap.add_argument('--verify-only',action='store_true');ap.add_argument('--weights-only',action='store_true');ap.add_argument('--component',default='',help='Optional weight path prefix, e.g. bundles/robocasa_mixed_sft10k');ap.add_argument('--report',type=Path);a=ap.parse_args()
 if not a.verify_only and not a.destination:ap.error('--destination required unless --verify-only')
 repo=Path(__file__).resolve().parents[1];weights=a.weights.resolve();manifest=json.loads((weights/'weights_manifest.json').read_text());expected=json.loads((repo/'migration/weights_manifest.json').read_text())
 if manifest!=expected:raise RuntimeError('Weight manifest differs from this code release')
 chosen=[r for r in manifest['files'] if r['path'].startswith(a.component)]
 if not chosen:raise ValueError('No selected weight files')
 checked=0;size=0
 for r in chosen:
  src=safe(weights,r['path'])
  if src.stat().st_size!=r['bytes'] or digest(src)!=r['sha256']:raise RuntimeError('Weight checksum mismatch: '+r['path'])
  if not a.verify_only:install(src,safe(a.destination.resolve(),r['restore_path']),r['sha256'])
  checked+=1;size+=r['bytes']
 code_count=0
 if not a.verify_only and not a.weights_only:
  source=repo/'research'
  for p in sorted(source.rglob('*')):
   if not p.is_file():continue
   install(p,safe(a.destination.resolve(),str(p.relative_to(source))),digest(p));code_count+=1
  # Preserve original internal aliases only where their target was actually restored.
  links=json.loads((repo/'migration/directory_links.json').read_text());old='/media/tyshao/'
  for r in links:
   link=safe(a.destination.resolve(),r['path'].removeprefix(old));target=safe(a.destination.resolve(),r['target'].removeprefix(old))
   if not target.exists():continue
   if link.exists() or link.is_symlink():
    if not link.is_symlink() or link.resolve()!=target.resolve():raise FileExistsError(link)
   else:link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=True)
 result=dict(state='VERIFIED' if a.verify_only else 'RESTORED',weight_files=checked,weight_bytes=size,code_files=code_count,component=a.component,training_data_included=False)
 if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
