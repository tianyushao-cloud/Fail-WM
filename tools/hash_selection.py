from pathlib import Path
import json,os,hashlib,time,concurrent.futures,collections
D=Path(__file__).resolve().parent;rows=json.loads((D/'selected_files.json').read_text())
# Reuse only hashes recorded for exactly the same inode,size,mtime_ns,ctime_ns.
cache={}
p=D/'hash_cache.jsonl'
if p.exists():
 for line in p.read_text().splitlines():
  try:r=json.loads(line);cache[tuple(r['key'])]=r['sha256']
  except ValueError:pass
start=time.monotonic();inode={};aliases={}
for r in rows:
 s=Path(r['path']).stat();k=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 r['key']=k;r['mode']=s.st_mode & 0o777
 if k not in inode:inode[k]=r
 aliases.setdefault(k,[]).append(r)
def work(pair):
 k,r=pair
 if k in cache:return k,cache[k]
 h=hashlib.sha256()
 with open(r['path'],'rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 s=Path(r['path']).stat()
 if k!=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns):raise RuntimeError('Source changed while hashing: '+r['path'])
 return k,h.hexdigest()
# Largest files first to give useful early size/progress estimates.
items=sorted(inode.items(),key=lambda x:-x[1]['bytes']);done=0;readbytes=0
with p.open('a',buffering=1) as log,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for k,h in pool.map(work,items):
  log.write(json.dumps(dict(key=k,sha256=h))+'\n');cache[k]=h
  for r in aliases[k]:r['sha256']=h
  done+=1;readbytes+=inode[k]['bytes']
  if done%100==0 or done==len(items):
   x=dict(done=done,total=len(items),hashed_gib=round(readbytes/2**30,2),elapsed_seconds=round(time.monotonic()-start,1))
   (D/'hash_progress.json').write_text(json.dumps(x));print(json.dumps(x),flush=True)
for r in rows:r.pop('key',None)
(D/'hashed_files.json').write_text(json.dumps(rows,indent=2))
unique={}
for r in rows:unique.setdefault(r['sha256'],r)
summary=dict(files=len(rows),unique_files=len(unique),logical_gib=round(sum(r['bytes'] for r in rows)/2**30,3),unique_gib=round(sum(r['bytes'] for r in unique.values())/2**30,3),elapsed_seconds=round(time.monotonic()-start,1))
(D/'dedup_summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',json.dumps(summary),flush=True)
