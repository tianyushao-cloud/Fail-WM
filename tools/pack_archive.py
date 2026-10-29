"""Build independently verifiable <2 GiB release parts; never modify experiment files."""
from pathlib import Path
import json,hashlib,tarfile,os,time,gzip,io,collections
import zstandard as zstd
D=Path(__file__).resolve().parent;OUT=D/'release_assets';OUT.mkdir(exist_ok=True)
rows=json.loads((D/'hashed_files.json').read_text());byhash={};groups=collections.defaultdict(list)
# The first occurrence is canonical. Each logical path is preserved in the manifest.
for r in rows:
 if r['sha256'] not in byhash:
  byhash[r['sha256']]=r;groups[r['group']].append(r)
manifest=[]
for r in rows:
 manifest.append(dict(path=r['relative_path'],sha256=r['sha256'],size=r['bytes'],mode=r['mode'],group=r['group'],archive_group=byhash[r['sha256']]['group']))
mp=OUT/'archive-manifest.jsonl.gz'
with mp.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
 for r in sorted(manifest,key=lambda x:x['path']):z.write((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode())
links=json.loads((D/'directory_links.json').read_text());base='/media/tyshao/'
(OUT/'directory-links.json').write_text(json.dumps([dict(path=x['path'].removeprefix(base),target=x['target'].removeprefix(base)) for x in links],indent=2))
class SplitWriter:
 def __init__(self,name):self.name=name;self.n=0;self.size=0;self.f=None;self.h=None;self.parts=[];self.limit=1900*2**20
 def writable(self):return True
 def flush(self):
  if self.f:self.f.flush()
 def finish(self):
  if self.f:
   self.f.flush();os.fsync(self.f.fileno());self.f.close();self.partial.replace(self.final)
   self.parts.append(dict(name=self.final.name,bytes=self.size,sha256=self.h.hexdigest()));self.f=None
 def write(self,b):
  total=len(b);v=memoryview(b)
  while v:
   if self.f is None:
    self.final=OUT/f'{self.name}.tar.zst.part{self.n:03d}';self.partial=Path(str(self.final)+'.partial')
    self.f=self.partial.open('wb');self.size=0;self.h=hashlib.sha256();self.n+=1
   take=min(len(v),self.limit-self.size);chunk=v[:take];self.f.write(chunk);self.h.update(chunk);self.size+=take;v=v[take:]
   if self.size==self.limit:self.finish()
  return total
class CheckedReader:
 def __init__(self,path):self.f=open(path,'rb');self.h=hashlib.sha256()
 def read(self,n=-1):
  b=self.f.read(n);self.h.update(b);return b
 def close(self):self.f.close()
index=dict(schema=1,repository='tianyushao-cloud/Fail-WM',tag='migration-20261029',format='sha256-addressed tar + zstd, split into 1900 MiB parts',groups={})
for group,rr in groups.items():
 receipt=OUT/f'{group}.receipt.json';planhash=hashlib.sha256(json.dumps([(r['sha256'],r['bytes']) for r in rr]).encode()).hexdigest()
 if receipt.exists():
  old=json.loads(receipt.read_text())
  if old['plan_sha256']==planhash and all((OUT/p['name']).stat().st_size==p['bytes'] for p in old['parts']):
   index['groups'][group]=old;print('REUSED',group,flush=True);continue
 writer=SplitWriter(group);started=time.monotonic();n=0
 print('PACK',group,len(rr),round(sum(r['bytes'] for r in rr)/2**30,3),'GiB',flush=True)
 with zstd.ZstdCompressor(level=3,threads=4).stream_writer(writer,closefd=False) as compressed:
  with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.USTAR_FORMAT) as tf:
   for r in rr:
    ti=tarfile.TarInfo(r['sha256']);ti.size=r['bytes'];ti.mode=0o644;ti.mtime=0
    reader=CheckedReader(r['path'])
    try:tf.addfile(ti,reader)
    finally:reader.close()
    if reader.h.hexdigest()!=r['sha256']:raise RuntimeError('Source changed since inventory: '+r['path'])
    n+=1
    if n%1000==0:(D/'pack_progress.json').write_text(json.dumps(dict(group=group,files=n,total=len(rr),seconds=round(time.monotonic()-started,1))))
 writer.finish()
 result=dict(unique_files=len(rr),logical_unique_bytes=sum(r['bytes'] for r in rr),plan_sha256=planhash,parts=writer.parts,seconds=round(time.monotonic()-started,1))
 receipt.write_text(json.dumps(result,indent=2));index['groups'][group]=result
 print('DONE',group,'parts',len(writer.parts),'GiB',round(sum(p['bytes'] for p in writer.parts)/2**30,3),flush=True)
 (OUT/'release-index.json').write_text(json.dumps(index,indent=2))
index['metadata']=[dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [mp,OUT/'directory-links.json']]
index['total_asset_bytes']=sum(x['bytes'] for g in index['groups'].values() for x in g['parts'])+sum(x['bytes'] for x in index['metadata'])
(OUT/'release-index.json').write_text(json.dumps(index,indent=2))
with (OUT/'SHA256SUMS').open('w') as f:
 for x in index['metadata']+[x for g in index['groups'].values() for x in g['parts']]:f.write(x['sha256']+'  '+x['name']+'\n')
(D/'pack_complete.json').write_text(json.dumps(dict(state='PACKED_NOT_UPLOADED',bytes=index['total_asset_bytes'],groups=len(groups)),indent=2))
print('ALL PACKED',index['total_asset_bytes']/2**30,'GiB',flush=True)
