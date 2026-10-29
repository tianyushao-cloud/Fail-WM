import json,csv,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;p=json.loads((R/'protocol_used.json').read_text());data=[]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
for row in p['cases']:
 d={n:json.loads((R/'results'/row['id']/n/'result.json').read_text()) for n in ['sft','rl20']}
 data.append(dict(id=row['id'],success=row['success'],failure_type=row.get('failure_type'),sft=d['sft'],rl20=d['rl20'],lpips_gain=d['sft']['metrics']['32']['lpips']-d['rl20']['metrics']['32']['lpips'],roi_lpips_gain=d['sft']['roi_metrics']['lpips']-d['rl20']['roi_metrics']['lpips'],track_error_reduction=d['sft']['object']['track_error_px_capped100']-d['rl20']['object']['track_error_px_capped100']))
summary={}
for group,rows in [('all',data),('success',[d for d in data if d['success']]),('failure',[d for d in data if not d['success']])]:
 out={'n':len(rows),'lpips_improved_cases':sum(d['lpips_gain']>0 for d in rows)}
 for name in ['sft','rl20']:
  out[name]={}
  for h in ['10','20','32']:out[name][h]={k:float(np.mean([d[name]['metrics'][h][k] for d in rows])) for k in ['lpips','ssim','psnr','mae','mse']}
  out[name]['roi']={k:float(np.mean([d[name]['roi_metrics'][k] for d in rows])) for k in ['lpips','ssim','psnr','mae']}
  out[name]['object']={k:float(np.mean([d[name]['object'][k] for d in rows])) for k in ['position','persistence','track_error_px_capped100','missing_rate','duplicate_rate','motion_error_px_capped50','false_lift_proxy_rate']}
 out['lpips32_relative_improvement_pct']=100*(out['sft']['32']['lpips']-out['rl20']['32']['lpips'])/out['sft']['32']['lpips'];summary[group]=out
summary['limitations']=['Fixed six-case exploratory validation subset, three initial groups; not a random population or held-out test estimate.','Three failure cases were previously inspected; cohort fixed before RL20 generation.','Only 32 future frames; late failure events may fall outside horizon.','Object metrics are 2D red-cube image proxies; position/persistence partly reuse training reward and are not independent validation.','Track error caps missing detections at 100px; motion error caps missing detections at 50px.','Same original actions, two initial observations, beam4, codec34k, float32, no GT history refresh.']
summary['per_case']=[{k:d[k] for k in ['id','success','failure_type','lpips_gain','roi_lpips_gain','track_error_reduction']} for d in sorted(data,key=lambda x:x['lpips_gain'],reverse=True)]
(R/'summary.json').write_text(json.dumps(summary,indent=2));(R/'paired_results.json').write_text(json.dumps(data,indent=2))
with (R/'per_case.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['id','success','failure_type','sft_lpips32','rl20_lpips32','lpips_gain','roi_lpips_gain','track_error_reduction_px'])
 for d in data:w.writerow([d['id'],d['success'],d['failure_type'],d['sft']['metrics']['32']['lpips'],d['rl20']['metrics']['32']['lpips'],d['lpips_gain'],d['roi_lpips_gain'],d['track_error_reduction']])
O=R/'videos';O.mkdir(exist_ok=True)
for d in data:
 row=next(x for x in p['cases'] if x['id']==d['id'])
 with np.load(Path(p['manifest']).parent/row['file']) as z:gt=z['images'][2:34].copy()
 movies=[gt]
 for name in ['sft','rl20']:
  with np.load(R/'results'/d['id']/name/'prediction.npz') as z:movies.append(np.rint(z['prediction']*255).clip(0,255).astype(np.uint8))
 frames=[]
 for t in range(32):
  im=Image.new('RGB',(960,600),'#151820');draw=ImageDraw.Draw(im)
  for col,(clip,name) in enumerate(zip(movies,['GROUND TRUTH','SFT10k','RL step20'])):
   x=320*col;frame=Image.fromarray(clip[t]);im.paste(frame,(x,40));draw.text((x+4,3),name,font=font,fill='white');draw.text((x+4,22),f'{d["id"]} t={t+2}',font=font,fill='yellow')
   if col:draw.text((x+4,300),f'LPIPS32: {d[["sft","rl20"][col-1]]["metrics"]["32"]["lpips"]:.4f}',font=font,fill='white')
   else:draw.text((x+4,300),'SUCCESS' if d['success'] else 'FAILURE',font=font,fill='white')
   im.paste(frame.crop((130,40,310,220)).resize((270,270)),(x+25,325))
  frames.append(np.asarray(im))
 target=O/f'{d["id"]}_GT_SFT_RL20.mp4'
 proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x600','-r','5','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
 proc.communicate(np.stack(frames).tobytes());assert proc.returncode==0
 sheet=Image.new('RGB',(960,4*300))
 for j,t in enumerate([9,17,25,31]):sheet.paste(Image.fromarray(frames[t]).resize((480,300)),((j%2)*480,(j//2)*300))
 sheet.crop((0,0,960,600)).save(O/f'{d["id"]}_keyframes.jpg',quality=95)
print(json.dumps({k:v for k,v in summary.items() if k!='limitations'},indent=2))
