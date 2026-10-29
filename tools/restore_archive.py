#!/usr/bin/env python3
"""Restore or independently verify the SHA256-addressed research archive.
Requires Python >=3.10 and zstandard. No original server paths are accessed.
"""
import argparse,collections,gzip,hashlib,io,json,os,shutil,tarfile
from pathlib import Path
import zstandard as zstd

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()

def safe(root,rel):
 p=Path(rel)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
 q=root/p
 if not q.resolve().is_relative_to(root.resolve()):raise ValueError('Destination escapes restore root')
 return q

class Parts(io.RawIOBase):
 def __init__(self,paths):super().__init__();self.paths=iter(paths);self.current=None
 def readable(self):return True
 def readinto(self,b):
  while True:
   if self.current is None:
    try:self.current=next(self.paths).open('rb')
    except StopIteration:return 0
   n=self.current.readinto(b)
   if n:return n
   self.current.close();self.current=None
 def close(self):
  if self.current:self.current.close()
  super().close()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--assets',type=Path,required=True)
 ap.add_argument('--destination',type=Path)
 ap.add_argument('--verify-only',action='store_true')
 ap.add_argument('--prefix',default='',help='Only restore paths beginning with this relative path prefix')
 ap.add_argument('--report',type=Path)
 a=ap.parse_args();assets=a.assets.resolve()
 if not a.verify_only and not a.destination:ap.error('--destination is required for restoring')
 root=(a.destination or Path('.')).resolve();index=json.loads((assets/'release-index.json').read_text())
 for r in index['metadata']:
  p=safe(assets,r['name'])
  if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise RuntimeError('Metadata checksum mismatch: '+p.name)
 with gzip.open(assets/'archive-manifest.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
 chosen=[r for r in rows if r['path'].startswith(a.prefix)];byhash=collections.defaultdict(list)
 for r in chosen:byhash[r['sha256']].append(r)
 groupnames=sorted({r['archive_group'] for r in chosen});seen=set();verified_bytes=0;restored=0
 for group in groupnames:
  info=index['groups'][group];paths=[]
  for r in info['parts']:
   p=safe(assets,r['name'])
   if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise RuntimeError('Part checksum mismatch: '+p.name)
   paths.append(p)
  print('VERIFY' if a.verify_only else 'RESTORE',group,flush=True)
  with io.BufferedReader(Parts(paths)) as stream,zstd.ZstdDecompressor().stream_reader(stream) as decompressed,tarfile.open(fileobj=decompressed,mode='r|') as tf:
   for ti in tf:
    if not ti.isfile() or not all(c in '0123456789abcdef' for c in ti.name) or len(ti.name)!=64:raise RuntimeError('Unexpected archive member')
    if ti.name not in byhash:continue
    expected=byhash[ti.name];first=expected[0]
    if ti.size!=first['size']:raise RuntimeError('Member length mismatch')
    h=hashlib.sha256();tmp=None;out=None
    if not a.verify_only:
     target=safe(root,first['path']);target.parent.mkdir(parents=True,exist_ok=True)
     if target.exists():
      if target.stat().st_size!=first['size'] or digest(target)!=ti.name:raise FileExistsError('Existing file differs: '+str(target))
     else:tmp=target.with_name(target.name+'.restore-partial');out=tmp.open('xb')
    try:
     with tf.extractfile(ti) as source:
      for b in iter(lambda:source.read(8*2**20),b''):
       h.update(b)
       if out:out.write(b)
    finally:
     if out:out.close()
    if h.hexdigest()!=ti.name:raise RuntimeError('Content checksum mismatch: '+ti.name)
    if tmp:tmp.chmod(first['mode']);tmp.replace(target)
    if not a.verify_only:
     for r in expected[1:]:
      dest=safe(root,r['path']);dest.parent.mkdir(parents=True,exist_ok=True)
      if dest.exists():
       if dest.stat().st_size!=r['size'] or digest(dest)!=ti.name:raise FileExistsError('Existing file differs: '+str(dest))
      else:shutil.copy2(target,dest);dest.chmod(r['mode'])
     restored+=len(expected)
    seen.add(ti.name);verified_bytes+=ti.size
 if set(byhash)!=seen:raise RuntimeError('Missing content blobs: '+str(len(set(byhash)-seen)))
 if not a.verify_only:
  for r in json.loads((assets/'directory-links.json').read_text()):
   link=safe(root,r['path']);target=safe(root,r['target'])
   if r['path'].startswith(a.prefix) and target.exists():
    if link.is_symlink():
     if link.resolve()!=target.resolve():raise FileExistsError(link)
    elif link.exists():raise FileExistsError(link)
    else:link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=True)
 result=dict(state='VERIFIED' if a.verify_only else 'RESTORED',files=len(chosen),unique_files=len(seen),verified_unique_bytes=verified_bytes,restored_files=restored,prefix=a.prefix)
 if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
