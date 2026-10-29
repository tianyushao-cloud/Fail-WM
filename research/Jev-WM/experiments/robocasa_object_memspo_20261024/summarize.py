import json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).parent
cfg=json.loads((R/'train.json').read_text());metrics=[json.loads(x) for x in (R/'train/metrics.jsonl').read_text().splitlines()]
partial='--partial' in sys.argv
assert (partial or len(metrics)==cfg['steps']) and len(metrics)>0
assert all(m['optimizer_updated'] and np.isfinite(m['grad_norm']) and m['grad_norm']>0 for m in metrics)
ck=R/'train/checkpoints'/f"step_{metrics[-1]['step']:06d}"
receipt=json.loads((ck/'experiment.json').read_text())
for name,want in receipt['files'].items():
 h=hashlib.sha256()
 with (ck/name).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==want,name
initial=Path(cfg['bundle'])/'robocasa/models/main/model.safetensors'
assert hashlib.sha256(initial.read_bytes()).hexdigest()!=receipt['files']['model.safetensors']
summary=dict(state='RL_PIPELINE_VERIFIED',steps=len(metrics),optimizer_updates=sum(m['optimizer_updated'] for m in metrics),informative_groups=sum(m['informative_groups'] for m in metrics),total_groups=len(metrics)*cfg['global_batch'],planned_steps=cfg['steps'],training_complete=(R/'train/complete.json').exists(),grad_norms=[m['grad_norm'] for m in metrics],checkpoint=str(ck),initial_weights_changed=True,checkpoint_files_verified=True,effectiveness='Not evaluated; this is a pipeline smoke run, not evidence of improvement.',metrics=metrics)
(R/('first_update_verified.json' if partial else 'summary.json')).write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
# Save a first-update candidate review only; it is an on-policy training example.
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
(R/'reviews').mkdir(exist_ok=True)
schedule=json.loads((R/'schedule.json').read_text())
for cursor in [0,1]:
 rec=json.loads((R/f'judge_log/{cursor:06d}.json').read_text())
 with np.load(R/f'judge_log/{cursor:06d}_clips.npz') as z:gt=z['reference'];pred=z['predictions']
 best=int(np.argmax(rec['raw_reward']));worst=int(np.argmin(rec['raw_reward']));frames=[]
 for t in range(len(gt)):
  im=Image.new('RGB',(960,296),'black');d=ImageDraw.Draw(im)
  for col,(arr,title) in enumerate([(gt[t],'GROUND TRUTH'),(pred[best,t],f'BEST candidate {best} R={rec["raw_reward"][best]:.3f}'),(pred[worst,t],f'WORST candidate {worst} R={rec["raw_reward"][worst]:.3f}')]):
   im.paste(Image.fromarray(arr),(col*320,40));d.text((col*320+4,2),title,font=font,fill='white');d.text((col*320+4,21),f'{schedule[cursor]["id"]} t={t+2}',font=font,fill='yellow')
  frames.append(np.asarray(im))
 path=R/f'reviews/group_{cursor:06d}_GT_best_worst.mp4'
 p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x296','-r','5','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
 p.communicate(np.stack(frames).tobytes());assert p.returncode==0
