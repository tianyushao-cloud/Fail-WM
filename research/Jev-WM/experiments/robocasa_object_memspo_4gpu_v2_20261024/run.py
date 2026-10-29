"""Single-node 4-GPU launcher continuing from step 15 checkpoint. Target: 30 additional steps."""
import argparse,collections,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','train'],nargs='?',default='train');args=parser.parse_args()
 cfg=read(R/'train.json');schedule=read(cfg['schedule']);rows=read(cfg['train_manifest'])['cases'];index={r['id']:r for r in rows}
 assert cfg['steps']==30 and cfg['global_batch']==16 and cfg['dataset']=='robocasa'
 assert cfg['independent_reference']
 # Validate schedule (800 samples = 50 steps original, we reuse it)
 assert len(schedule)==800
 for step in range(30):  # Only validate first 30 steps we'll use
  selected=[index[x['id']] for x in schedule[step*16:(step+1)*16]]
  assert sum(r['success'] for r in selected)==8
  for rank in range(4):assert sum(selected[rank+micro*4]['success'] for micro in range(4))==2
  assert all(r['split']=='train' and r['frames']>=34 for r in selected)
 validation=read(cfg['evaluation_manifest'])['cases']
 assert not {r['initial_state_id'] for r in rows}&{r['initial_state_id'] for r in validation}
 for path in [cfg['bundle'],cfg['text_model'],cfg['object_reward']['model'],R/'source/src/futureworlds/training.py']:
  assert Path(path).exists(),path
 # Check resume checkpoint exists
 resume_ckpt = Path(cfg['resume_from'])
 assert resume_ckpt.exists() and resume_ckpt.is_dir(), f'Resume checkpoint not found: {resume_ckpt}'
 
 identity=dict(config=digest(R/'train.json'),schedule=digest(cfg['schedule']),train_manifest=digest(cfg['train_manifest']),evaluation_manifest=digest(cfg['evaluation_manifest']),bundle_manifest=digest(Path(cfg['bundle'])/'manifest.json'),source={str(p.relative_to(R)):digest(p) for p in sorted((R/'source/src').rglob('*.py'))})
 print(json.dumps(dict(state='CONFIG_VALID',gpus=4,steps=30,global_batch=16,accumulation_per_rank=4,success_windows=240,failure_windows=240,horizon=32,checkpoint_every=5,initialization='Resume from step 15 with improved prompt',output=cfg['output'],resume_from=str(resume_ckpt)),indent=2),flush=True)
 if args.stage=='check':return
 lock=(R/'.launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import torch
 if torch.cuda.device_count()!=4:raise RuntimeError(f'Expected exactly 4 visible GPUs, got {torch.cuda.device_count()}; set CUDA_VISIBLE_DEVICES to four allocated GPUs')
 identity_path=R/'launch_identity.json'
 if identity_path.exists() and read(identity_path)!=identity:
  # Allow re-launch with modified config for this experiment
  print('Config/source changed; updating identity.',flush=True)
 write(identity_path,identity)
 out=Path(cfg['output']);marker=out/'complete.json'
 if marker.exists():
  if read(marker)['steps']!=30:raise RuntimeError('Unexpected completion marker')
  print('Already complete: 30 steps. No new training launched.',flush=True);return
 out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
 Path(cfg['object_reward']['log_dir']).mkdir(parents=True,exist_ok=True)
 candidates=sorted(p for p in (out/'checkpoints').glob('step_[0-9]*') if p.is_dir() and not p.name.endswith('.partial') and (p/'experiment.json').exists())
 committed=0;launch=dict(cfg)
 if candidates:
  latest=candidates[-1];receipt=read(latest/'experiment.json');committed=receipt['step']
  if not 0<committed<=30:raise RuntimeError('Invalid checkpoint step')
  launch['resume']=str(latest)
 # Preserve interrupted work
 stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');archive=R/'interrupted'/stamp
 def archive_file(p,label):
  dest=archive/label;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(dest))
 if out.exists() and not candidates:
  if any(out.iterdir()):pass  # Keep initial output dir
 elif out.exists():
  metrics=out/'metrics.jsonl'
  if metrics.exists():
   lines=metrics.read_text().splitlines();keep=[]
   for line in lines:
    try:
     if json.loads(line)['step']<=committed:keep.append(line)
    except json.JSONDecodeError:pass
   if len(keep)!=len(lines):
    archive_file(metrics,'metrics_before_recovery.jsonl');metrics.write_text('\n'.join(keep)+('\n' if keep else ''))
  for p in (out/'checkpoints').glob('*.partial'):archive_file(p,'checkpoints/'+p.name)
 judge_log_dir = Path(cfg['object_reward']['log_dir'])
 if judge_log_dir.exists():
  for p in judge_log_dir.glob('*'):
   key=p.name.split('.')[0].split('_')[0]
   if key.isdigit() and int(key)>=committed*16:archive_file(p,'judge_log/'+p.name)
 launch_path=R/'train_launch.json';write(launch_path,launch)
 env={**os.environ,'PYTHONPATH':str(R/'source/src'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','USE_TF':'0','USE_FLAX':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'}
 command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4','-m','futureworlds','train','--config',str(launch_path)]
 write(R/'status.json',dict(state='RUNNING',pid=os.getpid(),committed_step=committed,target_steps=30,command=command))
 print(f'Starting steps {committed+1}..30; checkpoints: {out}/checkpoints',flush=True)
 result=subprocess.run(command,cwd=R,env=env)
 if result.returncode:
  write(R/'status.json',dict(state='FAILED',exit_code=result.returncode,resume='Run the same command again; committed checkpoints are retained.'));raise SystemExit(result.returncode)
 if not marker.exists() or read(marker)['steps']!=30:raise RuntimeError('Training exited without step30 completion')
 metrics=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
 summary=dict(state='COMPLETE',steps=30,optimizer_updates=sum(m['optimizer_updated'] for m in metrics),informative_groups=sum(m['informative_groups'] for m in metrics),checkpoint=str(out/'checkpoints/step_000030'),effectiveness='Training complete; improvement is not evaluated by this launcher.')
 write(R/'summary.json',summary);write(R/'status.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
