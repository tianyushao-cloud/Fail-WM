import os,sys,json,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
sys.path.insert(0,str(R/'source/src'))
import numpy as np
import torch
from futureworlds.data import read_case,write_json
from futureworlds.exporting import export_model
from futureworlds.checkpoints import load_world_model,file_sha256
from futureworlds.pipeline import WorldPipeline
from futureworlds.evaluation import Metrics
from futureworlds.robocasa_object_reward import rule_scores,track

def object_metrics(gt,pred,initial):
 scores,details=rule_scores(gt,pred);gs,ps=details['gt'],details['pred'];errs=[];miss=0;dup=0;motion=[];visible=0
 anchor=track(initial[-1:])[0]['xy'];lift=0;eligible=0
 for g,p in zip(gs,ps):
  if g['xy'] is None:continue
  visible+=1;miss+=int(p['xy'] is None);dup+=int(p['count']>g['count'])
  errs.append(min(100.,float(np.linalg.norm(np.array(g['xy'])-p['xy']))) if p['xy'] is not None else 100.)
  if anchor is not None and np.linalg.norm(np.array(g['xy'])-anchor)<5:
   eligible+=1;lift+=int(p['xy'] is not None and p['xy'][1]<g['xy'][1]-20)
 for t in range(1,len(gs)):
  if gs[t]['xy'] is None or gs[t-1]['xy'] is None:continue
  if ps[t]['xy'] is None or ps[t-1]['xy'] is None:motion.append(50.)
  else:motion.append(min(50.,float(np.linalg.norm((np.array(ps[t]['xy'])-ps[t-1]['xy'])-(np.array(gs[t]['xy'])-gs[t-1]['xy'])))))
 return dict(**scores,track_error_px_capped100=float(np.mean(errs)) if errs else None,missing_rate=miss/max(1,visible),duplicate_rate=dup/max(1,visible),motion_error_px_capped50=float(np.mean(motion)) if motion else None,false_lift_proxy_rate=lift/max(1,eligible),false_lift_eligible_frames=eligible,visible_GT_frames=visible)

def main():
 p=json.loads((R/'protocol.json').read_text());torch.set_num_threads(2);torch.manual_seed(0)
 torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory)
 write_json(R/'status.json',dict(state='RUNNING',pid=os.getpid(),started=time.time(),completed_predictions=0,total_predictions=24))
 export_model(p['initial_bundle'],'robocasa',p['checkpoint'],R/'rl20_bundle')
 for n in ['robocasa/codec/diffusion_pytorch_model.safetensors','robocasa/preprocessing/action_ranges.pt']:
  assert file_sha256(Path(p['initial_bundle'])/n)==file_sha256(R/'rl20_bundle'/n)
 pipe=WorldPipeline(p['initial_bundle'],'robocasa','sft','cuda',p['text_model'])
 models={'sft':pipe.model,'rl20':load_world_model(R/'rl20_bundle','robocasa','main','cuda',verify=False)}
 metrics=Metrics('cuda');completed=0
 for row in p['cases']:
  images,actions,instruction=read_case(p['manifest'],row,52,32)
  for name,model in models.items():
   dest=R/'results'/row['id']/name;dest.mkdir(parents=True,exist_ok=True)
   if (dest/'result.json').exists():completed+=1;continue
   began=time.monotonic();pipe.model=model;torch.manual_seed(0)
   print(json.dumps(dict(event='generating',id=row['id'],model=name)),flush=True)
   with torch.inference_mode():
    generated=pipe.predict(images[:2],actions,instruction,32,4,'full')
    pred=generated['prediction'];values=metrics(pred,images[2:],[10,20,32]);roi=metrics(pred[:,40:220,130:310],images[2:,40:220,130:310],[32])
   rgb=np.rint(pred*255).clip(0,255).astype(np.uint8)
   obj=object_metrics(images[2:],rgb,images[:2])
   with (dest/'prediction.npz').open('wb') as f:np.savez_compressed(f,prediction=pred,tokens=generated['tokens'])
   result=dict(id=row['id'],model=name,success=row['success'],failure_type=row.get('failure_type'),seconds=time.monotonic()-began,**values,roi_metrics=roi['metrics']['32'],object=obj)
   write_json(dest/'result.json',result);completed+=1
   write_json(R/'status.json',dict(state='RUNNING',pid=os.getpid(),completed_predictions=completed,total_predictions=24,last_id=row['id'],last_model=name))
   print(json.dumps(dict(event='complete',id=row['id'],model=name,seconds=result['seconds'],metrics=values['metrics']['32'],object=obj)),flush=True)
 write_json(R/'status.json',dict(state='COMPLETE',completed_predictions=24,finished=time.time(),peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3))
if __name__=='__main__':
 try:main()
 except Exception as e:
  write_json(R/'status.json',dict(state='FAILED',error=repr(e)));traceback.print_exc();raise
