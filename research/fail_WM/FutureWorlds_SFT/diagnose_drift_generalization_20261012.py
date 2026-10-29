"""Matched train/val next-frame oracle versus open and late autonomous rollouts."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT');OUT=ROOT/'diagnostics_drift_generalization_20261012'
BUNDLE=ROOT/'evals_2x2_20261012/bundles/A'
RAW=Path('/media/tyshao/fail_WM/RoboCasa/data/batch1000/manifest.jsonl')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.partial');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def blobs(f):
 a=np.asarray(f,float);m=(a[...,0]>75)&(a[...,0]>1.5*a[...,1])&(a[...,0]>1.4*a[...,2]);lab,n=ndimage.label(m);v=[]
 for i in range(1,n+1):
  y,x=np.where(lab==i)
  if len(x)>=8:v.append(dict(x=float(x.mean()),y=float(y.mean()),area=len(x),box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]))
 return sorted(v,key=lambda v:-v['area'])
def roi(f):
 bs=blobs(f);m=np.zeros((256,320),bool)
 if bs:
  x0,y0,x1,y1=bs[0]['box'];m[max(0,y0-8):min(256,y1+8),max(0,x0-8):min(320,x1+8)]=True
 return m
def classify(f,rest):
 bs=blobs(f)
 table=any(abs(b['x']-rest[0])<=15 and abs(b['y']-rest[1])<=8 for b in bs)
 high=any(rest[1]-b['y']>15 for b in bs)
 return 'both' if table and high else 'table' if table else 'high' if high else 'other' if bs else 'none'
def select():
 rows={s:load(ROOT/'robocasa_data'/('train_natural.json' if s=='train' else 'val.json'))['cases'] for s in ('train','val')}
 raw={x['episode_id']:x for x in map(json.loads,RAW.read_text().splitlines())};used={'train':set(),'val':set()};cases=[];matches=[]
 def factors(r):return sorted(f['type'] for f in raw[r['id']]['perturbation']['factors'])
 def L(r):return raw[r['id']]['phase_ranges']['attempt0/lift']['first_action']//4+1
 def eligible(r,s,kind):
  if r['success'] or raw[r['id']]['event_summary']['lift'] or r['initial_state_id'] in used[s] or kind not in factors(r):return False
  with np.load(ROOT/'robocasa_data'/r['file']) as z:a=z['actions'].reshape(-1,13)
  phase=raw[r['id']]['phase_ranges']['attempt0/lift']
  return np.allclose(a[phase['first_action']:phase['last_action']+1,6],1) and r['frames']>L(r)+10 and any(x['success'] and x['initial_state_id']==r['initial_state_id'] and not factors(x) for x in rows[s])
 for kind in ['skip_descend']*3+['grasp_offset']*3:
  val=next(r for r in sorted(rows['val'],key=lambda r:r['id']) if eligible(r,'val',kind))
  train=min((r for r in rows['train'] if eligible(r,'train',kind) and factors(r)==factors(val)),key=lambda r:(abs(L(r)-L(val)),abs(r['frames']-val['frames']),r['id']))
  match=len(matches);matches.append(dict(match=match,factors=factors(val),train=train['id'],val=val['id'],train_L=L(train),val_L=L(val)))
  for split,fail in [('train',train),('val',val)]:
   used[split].add(fail['initial_state_id']);succ=next(r for r in sorted(rows[split],key=lambda r:r['id']) if r['success'] and r['initial_state_id']==fail['initial_state_id'] and not factors(r))
   for row in [fail,succ]:
    cases.append(dict(id=row['id'],split=split,role='success' if row['success'] else 'failure',match=match,group=row['initial_state_id'],L=L(row),end=L(row)+10,row=row))
 protocol=dict(state='FROZEN_BEFORE_NEW_INFERENCE',cases=cases,matches=matches,seed=20261012,beam=4,
  bundle_manifest_sha256=sha(BUNDLE/'manifest.json'),script_sha256=sha(__file__),raw_sha256=sha(RAW),
  manifests={s:sha(ROOT/'robocasa_data'/('train_natural.json' if s=='train' else 'val.json')) for s in rows},
  selection='val first3 distinct-group skip_descend and first3 distinct-group grasp_offset never-lift/+1; train matches exact perturbation-factor list then closest lift anchor, episode length, ID; same-group unperturbed success. No WM or codec outputs used to select.',
  modes=dict(start='GT first2 then free untilL+10',late='GT throughL+1 then freeL+2..L+10',oracle='GT anchor+recent6 refreshed independently at every targetL..L+10, free beam80tokens per next frame'),
  statistics='Compare same-case late/oracle on common L+2..L+10; compare split within six matched strata. Group/stratum bootstrap exploratory, n=6 not independent generated frames.',
  semantics='high is red component >15px above resting cube; both/none/other explicit; original rise threshold21.166... from frozen val selection also reported; successes use endpoint high-only and window lift, not task completion',
  limitations=['24 cases/12 initial groups/6matched strata; train frames are in-sample, val is previously inspected; no test/generalization population estimate','Matching factor/phase does not match all physical/image difficulty; gap is evidence consistent with generalization, not proof of overfit','Oracle vs free varies all generated history including arm/cube; does not isolate cube-only cause'])
 p=OUT/'protocol.json'
 if p.exists():assert load(p)==protocol
 else:write(p,protocol)
 print(json.dumps(matches),flush=True)

def run():
 import torch
 from futureworlds.pipeline import WorldPipeline
 from futureworlds.data import read_case
 from futureworlds.memory import make_prefix,retained
 from futureworlds.decoding import beam_decode
 from futureworlds.batch import TensorBatch
 protocol=load(OUT/'protocol.json');assert protocol['script_sha256']==sha(__file__)
 assert protocol['bundle_manifest_sha256']==sha(BUNDLE/'manifest.json')
 torch.set_num_threads(2);torch.manual_seed(20261012)
 torch.cuda.set_per_process_memory_fraction(16*1024**3/torch.cuda.get_device_properties(0).total_memory)
 pipe=WorldPipeline(BUNDLE,'robocasa','sft','cuda','/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base');pipe.model.eval()
 selection=load(ROOT/'evals_gt_handoff_20261010/selection.json');threshold=selection['rise_threshold_px']
 @torch.inference_mode()
 def nll(prefix,target):
  target=target.long();logits=pipe.model(input_ids=torch.cat([prefix,target[:,:-1]],1),use_cache=False).logits[:,prefix.shape[1]-1:].float()
  return (-logits.log_softmax(-1).gather(-1,target.unsqueeze(-1)).squeeze()).cpu().numpy()
 for case in protocol['cases']:
  cid,split,L,end=case['id'],case['split'],case['L'],case['end'];dest=OUT/'cases'/split/cid;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'result.json').exists():continue
  began=time.time();manifest=ROOT/'robocasa_data'/('train_natural.json' if split=='train' else 'val.json')
  assert sha(manifest)==protocol['manifests'][split]
  images,actions,text=read_case(manifest,case['row'],52)
  with torch.inference_mode():ctx,dyn,acts,_,_=pipe.encode_training_window(images,actions,text);pipe.condition(text)
  context=ctx.reshape(1,-1)+4375
  rest=next((blobs(images[t])[0] for t in range(L-1,-1,-1) if blobs(images[t])),None)
  assert rest is not None;rest=[rest['x'],rest['y']]
  def prefix(history,t):return make_prefix(context,history,acts,t)[0]
  @torch.inference_mode()
  def decode(codes,k):
   s=k-k%2;full=torch.cat([dyn[s:k][None],codes],1)
   pix=pipe.worker.detokenize(TensorBatch({'tokens':full,'ctx_tokens':ctx}),None).batch['pixels'][0,1+k-s:]
   return pix.permute(0,2,3,1).cpu().numpy()*255
  recon=decode(dyn[L:end+1][None],L);movies={'gt':images[L:end+1].astype(float),'codec':recon};metrics={};tokens={};reuse={}
  for mode,k in [('oracle',L),('start',2),('late',L+2)]:
   cache_path=dest/f'{mode}.npz'
   if cache_path.exists():
    with np.load(cache_path) as z:generated=z['tokens'];losses=z['nll'];movie=z['pixels']
    reuse[mode]='own checkpoint'
   else:
    existing=None
    if split=='val' and mode!='oracle':
     res=ROOT/'evals_gt_handoff_codec34k_20261012/results/c34_natural'/f'{cid}.json'
     if res.exists():
      rr=load(res);receipt=load(res.parents[2]/'run_c34_natural.json')
      oldbundle=Path(receipt['models']['c34_natural']['bundle'])
      assert load(oldbundle/'manifest.json')['training_checkpoint_sha256']==load(BUNDLE/'manifest.json')['training_checkpoint_sha256']
      assert rr['anchor']==L and rr['end']==end
      h=next((h for h in rr['handoffs'] if h['k']==k),None)
      if h:
       with np.load(res.with_name(cid+'_tokens.npz')) as z:existing=z[f'k{k:03d}'][0].astype(np.int64)
       assert existing.shape==(end-k+1,80);reuse[mode]=str(res)
    history={i:dyn[i:i+1] for i in retained(k)};generated=[];losses=[]
    for t in range(k,end+1):
     if mode=='oracle':history={i:dyn[i:i+1] for i in retained(t)}
     pref=prefix(history,t)
     if t>=L:losses.append(nll(pref,dyn[t:t+1]))
     if existing is None:
      with torch.inference_mode():cand,_,_=beam_decode(pipe.model,pref,width=4)
      code=cand[:1]
     else:code=torch.as_tensor(existing[t-k:t-k+1],device='cuda')
     history[t]=code;generated.append(code[0].cpu().numpy());history={i:v for i,v in history.items() if i==1 or i>=t-5}
    generated=np.stack(generated);losses=np.stack(losses)
    movie=decode(torch.as_tensor(generated,device='cuda')[None],k)
    with cache_path.with_suffix('.partial').open('wb') as f:np.savez_compressed(f,tokens=generated,nll=losses,pixels=movie)
    cache_path.with_suffix('.partial').replace(cache_path)
   movies[mode]=movie[max(L-k,0):];tokens[mode]=generated
   per=[]
   for j,t in enumerate(range(max(L,k),end+1)):
    gt=images[t];pred=movies[mode][j];mask=roi(gt);tile=mask.reshape(8,32,10,32).any((1,3)).reshape(80)
    err=np.abs(pred-gt).mean(-1);cls=classify(pred,rest);bs=blobs(pred)
    per.append(dict(t=t,offset=t-L,history_steps_generated=0 if mode=='oracle' else t-k,
       nll=float(losses[j].mean()),roi_nll=float(losses[j][tile].mean()) if tile.any() else None,
       mae=float(err.mean()),roi_mae=float(err[mask].mean()) if mask.any() else None,
       outside_mae=float(err[~mask].mean()),state=cls,gt_state=classify(gt,rest),
       high21=any(rest[1]-b['y']>threshold for b in bs),components=bs))
   metrics[mode]=per
  result=dict(case=case,rest=rest,threshold=threshold,metrics=metrics,reused=reuse,
    gt=[dict(t=L+i,state=classify(f,rest),components=blobs(f)) for i,f in enumerate(movies['gt'])],
    codec=[dict(t=L+i,state=classify(f,rest),components=blobs(f)) for i,f in enumerate(movies['codec'])],
    seconds=time.time()-began)
  # Compact views focus on object and gripper, retain full predictions in NPZ.
  canvas=Image.new('RGB',(5*180,9*196),'white');draw=ImageDraw.Draw(canvas)
  for row,t in enumerate(range(L+2,end+1)):
   for col,mode in enumerate(['gt','codec','oracle','late','start']):
    f=movies[mode][t-(L+2 if mode=='late' else L)]
    crop=Image.fromarray(np.uint8(np.clip(f,0,255))).crop((130,40,310,220))
    canvas.paste(crop,(col*180,row*196+16));draw.text((col*180+2,row*196+2),f'{mode} t{t} L+{t-L}',fill='black')
  canvas.save(dest/'comparison.jpg',quality=92);write(dest/'result.json',result)
  print(json.dumps(dict(split=split,id=cid,role=case['role'],seconds=result['seconds'])),flush=True)
 write(OUT/'complete.json',dict(state='COMPLETE',cases=len(protocol['cases']),peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','run']);a=p.parse_args();OUT.mkdir(exist_ok=True)
 {'select':select,'run':run}[a.phase]()
