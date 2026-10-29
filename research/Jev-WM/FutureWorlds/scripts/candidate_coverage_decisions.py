"""Paired 20-initial-state calibration: four versus eight grouped beam candidates.
No training, no threshold tuning, no evaluation/test split access.
"""
import argparse, concurrent.futures, gc, hashlib, json, os, threading, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from futureworlds.data import read_case, read_manifest
from futureworlds.decision_reward import write_json, atomic_write, canonical
from futureworlds.checkpoints import file_sha256
from futureworlds.protocol import implementation
from futureworlds.focused_alignment import FocusedStore, PROMPT, QUESTIONS, crop_box, rgb
from futureworlds.reliable_pair import select_pair, POLICY, COMPONENTS
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/"runs/local_success_decisions/candidate_coverage"
PRIOR=ROOT/"runs/local_success_decisions"
VARIANTS={"four":dict(groups=4,beams=2,penalty=.1),"eight":dict(groups=8,beams=2,penalty=.1)}
SEED=20261021
MAX_CALLS=480
def load(p):return json.loads(Path(p).read_text())
def log(**kw):print(json.dumps(kw),flush=True)
def signature():
 return dict(implementation=implementation(),runner_sha256=file_sha256(__file__))
def prepare():
 EXP.mkdir(exist_ok=True)
 if (EXP/"protocol.json").exists():
  p=load(EXP/"protocol.json")
  if p["signature"]!=signature():raise RuntimeError("Implementation changed; protocol cannot be reused")
  return p
 cfg=load(PRIOR/"reliable_pair/train.json")
 manifest=read_manifest(cfg["train_manifest"],"robocasa")
 excluded=set()
 # All previously generated calibration episodes, plus reserved protocol groups.
 for p in PRIOR.rglob("candidates.npz"):
  name=p.parent.name
  if name.startswith("g") and "_b" in name:excluded.add("counter_group_"+name[1:6])
 def visit(value):
  if isinstance(value,dict):
   if "initial_state_id" in value:excluded.add(value["initial_state_id"])
   for k,v in value.items():
    if k in ("calibration_rows",):visit(v)
  elif isinstance(value,list):
   for v in value:visit(v)
 for p in PRIOR.rglob("*protocol.json"):
  if EXP in p.parents:continue
  visit(load(p))
 groups={}
 for row in manifest["cases"]:
  if row["initial_state_id"] not in excluded:
   if not row["success"] or row["source_start"]!=0 or row["split"]!="train":raise ValueError("Invalid source row")
   groups.setdefault(row["initial_state_id"],[]).append(row)
 if len(groups)<20:raise ValueError("Not enough unseen training groups")
 rng=np.random.default_rng(SEED)
 selected=rng.permutation(sorted(groups))[:20]
 rows=[groups[g][int(rng.integers(len(groups[g])))] for g in selected]
 p=dict(seed=SEED,rows=rows,excluded_initial_states=sorted(excluded),variants=VARIANTS,
  source_bundle=cfg["bundle"],source_bundle_sha256=file_sha256(Path(cfg["bundle"])/"manifest.json"),
  train_manifest=cfg["train_manifest"],train_manifest_sha256=file_sha256(cfg["train_manifest"]),
  text_model=cfg["text_model"],horizon=32,initial_frames=[0,1],selection=POLICY,prompt=PROMPT,questions=QUESTIONS,
  model="gpt-6-luna",max_api_attempts=MAX_CALLS,spacing_seconds=3,retries=0,
  primary_endpoint="paired difference in fraction of scenes with an admissible candidate pair",
  secondary_endpoints=["unique trajectories","mean pairwise token difference","selected winner minus loser LPIPS",
                       "selected winner LPIPS versus mean and best candidate","fraction of zero/one/two valid variants"],
  review="Blinded A/B sheets of all 32 frames; mapping sealed separately; review after scoring. Developer visual review is not independent human annotation.",
  interpretation="Calibration only. More eligible pairs can reflect multiple comparisons; neither rate nor LPIPS proves semantic accuracy. No automatic RL launch.",
  signature=signature())
 write_json(EXP/"protocol.json",p)
 log(event="protocol_frozen",rows=[r["id"] for r in rows],excluded=len(excluded))
 return p
def runtime():
 torch.set_num_threads(2)
 torch.cuda.set_per_process_memory_fraction(.25,0)
 os.environ["OPENAI_API_KEY"]=Path("/root/.config/jev-wm/openai.key").read_text().strip()
 for k in ("HTTPS_PROXY","HTTP_PROXY","https_proxy","http_proxy"):os.environ[k]="http://127.0.0.1:17897"
 os.environ["NO_PROXY"]=os.environ["no_proxy"]="127.0.0.1,localhost"
class Budget:
 """Count attempts across restarts; write before request, including failed attempts."""
 def __init__(self,client):
  self.client=client;self.path=EXP/"attempts.jsonl"
  self.used=sum(1 for _ in self.path.open()) if self.path.exists() else 0
 def score(self,payload):
  if self.used>=MAX_CALLS:raise RuntimeError("Persistent API attempt budget exhausted")
  self.used+=1
  with self.path.open("a") as f:
   f.write(json.dumps(dict(attempt=self.used,time=time.time(),payload_sha256=hashlib.sha256(canonical(payload).encode()).hexdigest()))+"\n")
   f.flush();os.fsync(f.fileno())
  return self.client.score(payload)
def render(folder,z,choice):
 if choice is None:return
 rng=np.random.default_rng(int(hashlib.sha256(str(folder).encode()).hexdigest()[:8],16))
 win,lose=choice["winner"],choice["loser"]
 mapping=[win,lose] if rng.integers(2) else [lose,win]
 write_json(folder/"blind_mapping.json",dict(A=mapping[0],B=mapping[1],winner=win,loser=lose))
 box=crop_box(z["initial"],z["reference"])
 # Four readable sheets, eight timestamps each, with full scene and common detail.
 for block in range(4):
  canvas=Image.new("RGB",(8*240,3*400),"white");draw=ImageDraw.Draw(canvas)
  for row,(label,seq) in enumerate([("REFERENCE",z["reference"]),("A",z["prediction"][mapping[0]]),("B",z["prediction"][mapping[1]])]):
   for col in range(8):
    t=block*8+col;x=col*240;y=row*400
    full=ImageOps.contain(Image.fromarray(seq[t]),(236,175),Image.Resampling.NEAREST)
    detail=ImageOps.contain(Image.fromarray(seq[t]).crop(box),(236,195),Image.Resampling.NEAREST)
    canvas.paste(full,(x,y+24));canvas.paste(detail,(x,y+200))
    draw.text((x+5,y+5),f"{label} t={t+2}",fill="black")
  canvas.save(folder/f"blind_{block}.jpg",quality=92)
def summarize(protocol):
 results=[]
 for row in protocol["rows"]:
  pairs={}
  for name in VARIANTS:
   p=EXP/"cases"/row["id"]/name/"selection.json"
   if p.exists():pairs[name]=load(p)
  results.append(dict(id=row["id"],variants=pairs))
 completed=[r for r in results if len(r["variants"])==2]
 summary=dict(state="COMPLETE" if len(completed)==20 else "RUNNING",completed_pairs=len(completed),target_pairs=20,
              variants={},paired=None,signature=signature(),protocol_sha256=file_sha256(EXP/"protocol.json"))
 for name in VARIANTS:
  available=[r["variants"][name] for r in results if name in r["variants"]]
  valid=[r for r in available if r["selection"]["chosen"]]
  summary["variants"][name]=dict(scored_scenes=len(available),eligible_scenes=len(valid),
   eligible_rate=len(valid)/len(available) if available else None,
   mean_unique_trajectories=float(np.mean([r["diversity"]["unique_trajectories"] for r in available])) if available else None,
   mean_pairwise_token_difference=float(np.mean([r["diversity"]["mean_pairwise_token_difference_fraction"] for r in available])) if available else None,
   selected_lpips_mean=float(np.mean([r["selected_lpips"] for r in valid])) if valid else None,
   selected_minus_loser_lpips_mean=float(np.mean([r["selected_lpips"]-r["loser_lpips"] for r in valid])) if valid else None)
 if completed:
  a=np.array([bool(r["variants"]["four"]["selection"]["chosen"]) for r in completed],dtype=int)
  b=np.array([bool(r["variants"]["eight"]["selection"]["chosen"]) for r in completed],dtype=int)
  diff=b-a;rng=np.random.default_rng(SEED+1)
  bootstrap=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(axis=1)
  summary["paired"]=dict(rate_difference_eight_minus_four=float(diff.mean()),bootstrap_95_percent_ci=np.quantile(bootstrap,[.025,.975]).tolist(),
   both=int(((a==1)&(b==1)).sum()),four_only=int(((a==1)&(b==0)).sum()),eight_only=int(((a==0)&(b==1)).sum()),neither=int(((a==0)&(b==0)).sum()))
 summary["api_attempts"]=sum(1 for _ in (EXP/"attempts.jsonl").open()) if (EXP/"attempts.jsonl").exists() else 0
 summary["rl_updates"]=0
 write_json(EXP/"summary.json",summary)
 write_json(EXP/"results.json",results)
 lines=["# 20场景：4候选与8候选对照","",
        f"状态：{summary['state']}，完成 {len(completed)}/20 个配对场景；API尝试 {summary['api_attempts']}/{MAX_CALLS}。RL更新0。",
        "","提示词、组件评分与可靠对门槛固定；原始首两帧条件、32帧预测、成功训练数据。","",
        "|方案|已评分场景|有可靠对场景|有效率|平均不同轨迹数|","|---|---:|---:|---:|---:|"]
 for name,s in summary["variants"].items():
  lines.append(f"|{name}|{s['scored_scenes']}|{s['eligible_scenes']}|{s['eligible_rate']}|{s['mean_unique_trajectories']}|")
 if summary["paired"]:lines+=["","配对结果：",json.dumps(summary["paired"],ensure_ascii=False)]
 lines+=["","这些是候选覆盖率，不是RL涨点。多候选也增加误选机会，必须结合盲化图检查语义正确性。",
         "LPIPS只作离线诊断，不进入奖励。不得只因候选与LPIPS一致，就认定物体行为正确。",
         "API回执见 reviews/；逐例指标和选择见 cases/<id>/<four|eight>/；blind_0..3.jpg 展示全部32帧，blind_mapping.json 为对应关系。",
         "本实验只比较候选数，groups=4/8，beams=2，penalty=0.1，其余不变。没有自动启动训练。"]
 (EXP/"REPORT.md").write_text("\n".join(lines)+"\n")
 return summary
def score_case(protocol,store,row,name):
 folder=EXP/"cases"/row["id"]/name
 meta=load(folder/"metadata.json")
 if meta["signature"]!=signature() or file_sha256(folder/"candidates.npz")!=meta["sha256"]:raise RuntimeError("Candidate provenance mismatch")
 path=folder/"selection.json"
 if path.exists():
  s=load(path)
  if s["candidate_sha256"]!=meta["sha256"] or s["protocol_sha256"]!=file_sha256(EXP/"protocol.json"):raise RuntimeError("Selection provenance mismatch")
  return
 with np.load(folder/"candidates.npz",allow_pickle=False) as z:
  components=[];keys=[]
  for reverse in [False,True]:
   store.prediction_first=reverse;rows=[];orderkeys=[]
   for i,p in enumerate(z["prediction"]):
    key,_=store.score_arrays(z["initial"],p,z["actions"][2:],str(z["instruction"].item()),"robocasa",z["reference"])
    c=store.components(key);rows.append([c[n] for n in COMPONENTS]);orderkeys.append(key)
    log(event="scored",id=row["id"],variant=name,candidate=i,reverse=reverse,attempts=store.client.used)
   components.append(rows);keys.append(orderkeys)
  rewards,selection=select_pair(components);chosen=selection["chosen"];metrics=meta["metrics"]
  result=dict(id=row["id"],variant=name,selection=selection,rewards=rewards.tolist(),components=components,keys=keys,
   metrics=metrics,diversity=meta["trace"],candidate_sha256=meta["sha256"],protocol_sha256=file_sha256(EXP/"protocol.json"),
   selected_lpips=metrics[chosen["winner"]]["lpips"] if chosen else None,
   loser_lpips=metrics[chosen["loser"]]["lpips"] if chosen else None)
  render(folder,z,chosen)
  write_json(path,result)
 log(event="selection",id=row["id"],variant=name,chosen=chosen)
 summarize(protocol)
def collect_case(protocol,pipe,worker,metric,row,name):
 folder=EXP/"cases"/row["id"]/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/"metadata.json").exists():
  m=load(folder/"metadata.json")
  if m["signature"]!=signature() or m["sha256"]!=file_sha256(folder/"candidates.npz"):raise RuntimeError("Candidate cache changed")
  return
 from futureworlds.search import rollout_group_beam
 from futureworlds.batch import TensorBatch
 started=time.monotonic();images,actions,task=read_case(protocol["train_manifest"],row,pipe.action_dim,32)
 torch.manual_seed(SEED)
 with torch.no_grad():
  ctx,dyn,acts,pixels,cond=pipe.encode_training_window(images,actions,task)
  frames,trace=rollout_group_beam(pipe.model,ctx.reshape(1,-1)+4375,dyn[1:2],acts,32,**VARIANTS[name])
  codes=torch.stack([f["response"] for f in frames],1)
  # Chunk codec decoding to bound peak memory for eight candidates.
  decoded=[]
  for start in range(0,len(codes),2):
   sub=codes[start:start+2]
   pred=worker.detokenize(TensorBatch(dict(tokens=sub,ctx_tokens=ctx.expand(len(sub),-1,-1))),None).batch["pixels"][:,1:]
   decoded.extend(rgb(p) for p in pred)
  predictions=np.stack(decoded)
  metrics=[metric(p.astype(np.float32)/255,images[2:],[32])["metrics"]["32"] for p in predictions]
 atomic_write(folder/"candidates.npz",lambda f:np.savez_compressed(f,initial=images[:2],reference=images[2:],prediction=predictions,
                                                  actions=actions,instruction=np.array(task),tokens=codes.cpu().numpy()))
 write_json(folder/"metadata.json",dict(id=row["id"],variant=name,signature=signature(),sha256=file_sha256(folder/"candidates.npz"),
   metrics=metrics,trace=trace,seconds=time.monotonic()-started))
 log(event="generated",id=row["id"],variant=name,seconds=time.monotonic()-started,unique=trace["unique_trajectories"])
def run():
 protocol=prepare();runtime()
 if file_sha256(protocol["train_manifest"])!=protocol["train_manifest_sha256"]:raise RuntimeError("Manifest changed")
 if file_sha256(Path(protocol["source_bundle"])/"manifest.json")!=protocol["source_bundle_sha256"]:raise RuntimeError("Bundle changed")
 store=FocusedStore(dict(backend="api",model=protocol["model"],review_dir=str(EXP/"reviews"),max_api_calls=MAX_CALLS,
                         timeout_seconds=90,retries=0,spacing_seconds=3))
 store.client=Budget(store.client)
 from futureworlds.pipeline import WorldPipeline
 from futureworlds.rewards import CodecWorker
 from futureworlds.evaluation import Metrics
 pipe=WorldPipeline(protocol["source_bundle"],"robocasa","sft","cuda",protocol["text_model"])
 worker=CodecWorker(pipe.codec,None);metric=Metrics("cuda")
 summarize(protocol)
 # One GPU producer, one API consumer. Only API consumer touches its quota/cache.
 with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
  jobs=[]
  for index,row in enumerate(protocol["rows"]):
   for name in (["four","eight"] if index%2==0 else ["eight","four"]):
    for job in jobs:
     if job.done():job.result()
    collect_case(protocol,pipe,worker,metric,row,name)
    jobs.append(pool.submit(score_case,protocol,store,row,name))
  del pipe,worker,metric;gc.collect();torch.cuda.empty_cache()
  for job in jobs:job.result()
 summary=summarize(protocol)
 write_json(EXP/"status.json",dict(state="COMPLETE",completed_pairs=summary["completed_pairs"],rl_updates=0,time=time.time()))
 log(event="complete",summary=summary)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","run","summarize"]);a=p.parse_args()
 try:
  if a.stage=="prepare":prepare()
  elif a.stage=="summarize":print(json.dumps(summarize(prepare()),indent=2))
  else:run()
 except Exception as exc:
  # Error class only: transport responses/credentials never go to report.
  write_json(EXP/"status.json",dict(state="STOPPED",error_type=type(exc).__name__,time=time.time(),rl_updates=0))
  log(event="stopped",error_type=type(exc).__name__)
  raise

