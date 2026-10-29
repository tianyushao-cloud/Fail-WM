
import concurrent.futures,hashlib,json,pathlib,subprocess,time,urllib.request
ROOT=pathlib.Path("/media/tyshao/Jev-WM/paper_assets/futureworlds_rt1")
AUDIT=ROOT.parent/"audit_rt1"
REV="64b9802cb328b66c8ea263f2a4acd795c28bb144"
BASE="https://huggingface.co/easylearning/FutureWorlds/resolve/"+REV+"/"
PROXY="http://192.168.32.28:18000"
ROOT.mkdir(parents=True,exist_ok=True);AUDIT.mkdir(parents=True,exist_ok=True)
op=urllib.request.build_opener(urllib.request.ProxyHandler({"https":PROXY}))
def fetch(url):
 for attempt in range(4):
  try:
   with op.open(url,timeout=60) as r:return r.read()
  except Exception:
   if attempt==3:raise
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
raw=fetch(BASE+"manifest.json")
(AUDIT/"upstream_manifest.json").write_bytes(raw)
manifest=json.loads(raw)
files=[x for x in manifest["files"] if x["path"].startswith("rt1/")]
meta=json.loads(fetch("https://huggingface.co/api/models/easylearning/FutureWorlds/revision/"+REV))
(AUDIT/"upstream_model_info.json").write_text(json.dumps(meta,indent=2))
print("DOWNLOAD",len(files),"files",sum(x["bytes"] for x in files),"bytes",flush=True)
def download(item):
 rel=pathlib.PurePosixPath(item["path"])
 assert not rel.is_absolute() and ".." not in rel.parts
 p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  assert p.stat().st_size==item["bytes"] and sha(p)==item["sha256"],str(p)
  print("VERIFIED_EXISTING",rel,flush=True);return
 tmp=p.with_name(p.name+".partial")
 for attempt in range(4):
  result=subprocess.run(["curl","--proxy",PROXY,"--fail","--location","--silent","--show-error","--retry","3","--retry-delay","2","--connect-timeout","30","--max-time","1800","--continue-at","-","--output",str(tmp),BASE+str(rel)],capture_output=True,text=True)
  if result.returncode==0 and tmp.stat().st_size==item["bytes"] and sha(tmp)==item["sha256"]:
   tmp.replace(p);print("VERIFIED",rel,item["bytes"],flush=True);return
  print("RETRY",rel,"attempt",attempt+1,"exit",result.returncode,"bytes",tmp.stat().st_size if tmp.exists() else 0,flush=True)
  if tmp.exists() and tmp.stat().st_size>=item["bytes"]:tmp.unlink()
 raise RuntimeError("Failed download/verification: "+str(rel))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 list(pool.map(download,files))
for name in ["README.md","weights_manifest.json"]:
 (ROOT/name).write_bytes(fetch(BASE+name))
for item in json.loads((ROOT/"weights_manifest.json").read_text())["components"]:
 for f in item["files"]:
  if f["path"].startswith("rt1/") and f.get("hash_scope")=="release_file":
   assert sha(ROOT/f["path"])==f["sha256"],f["path"]
subset={"schema_version":1,"source_repo":"easylearning/FutureWorlds","source_revision":REV,"source_manifest_sha256":hashlib.sha256(raw).hexdigest(),"files":files}
(ROOT/"manifest.json").write_text(json.dumps(subset,indent=2)+"\n")
report={"status":"COMPLETE","repo":"easylearning/FutureWorlds","revision":REV,"bundle":str(ROOT),"verified_files":len(files),"verified_bytes":sum(x["bytes"] for x in files),"all_sha256_match":True,"variants":["sft","memspo200","grpo200","ordinary200","main"]}
(AUDIT/"weights_verification.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)
