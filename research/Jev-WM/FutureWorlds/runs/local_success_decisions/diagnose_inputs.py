
import json,hashlib,base64,io
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from futureworlds.reference_alignment import ReferenceAlignmentStore
from futureworlds.decision_api import normalized_scores
from futureworlds.evaluation import Metrics
torch.set_num_threads(2)
root=Path('/media/tyshao/Jev-WM/FutureWorlds'); exp=root/'runs/local_success_decisions'
out=exp/'diagnosis';out.mkdir(exist_ok=True)
r=json.loads((exp/'calibration.json').read_text());protocol=json.loads((exp/'protocol.json').read_text())
cfg=json.loads((exp/'train.json').read_text()); m=json.loads(Path(cfg['train_manifest']).read_text()); index={row['id']:row for row in m['cases']}
metric=Metrics('cpu');result=[]
for group in r['results']:
 folder=exp/'calibration'/group['id']
 with np.load(folder/'candidates.npz') as z:
  init,truth,preds=z['initial'],z['reference'],z['prediction']
  with np.load(Path(cfg['train_manifest']).parent/index[group['id']]['file']) as src:
   assert np.array_equal(init,src['images'][:2]);assert np.array_equal(truth,src['images'][2:34])
  case=dict(id=group['id'],candidates=[],controls={})
  for label,value in group['scores'].items():
   packet=json.loads((exp/'reviews'/value['key']/'request.json').read_text())
   receipt=json.loads((exp/'reviews'/value['key']/'score.json').read_text())
   scores=normalized_scores(receipt['official_response'],packet['questions'],packet['model'])
   assert abs(sum(scores.values())/3-value['score'])<1e-10
   with np.load(exp/'reviews'/value['key']/'frames.npz') as arrays:
    for name,meta in packet['arrays'].items():
     assert hashlib.sha256(arrays[name].tobytes()).hexdigest()==meta['sha256']
    assert np.array_equal(arrays['reference_rgb'],truth)
    if label.startswith('candidate_'):assert np.array_equal(arrays['prediction_rgb'],preds[int(label[-1])])
    payload=ReferenceAlignmentStore.build_payload(packet,arrays)
    parts=[p for p in payload['input'][0]['content'] if p['type']=='input_image']
    assert len(parts)==8
    for part,off in zip(parts,packet['sampled_prediction_offsets']):
     pixels=np.array(Image.open(io.BytesIO(base64.b64decode(part['image_url'].split(',')[1]))))
     assert np.array_equal(pixels[:,:320],truth[off])
     assert np.array_equal(pixels[:,320:],arrays['prediction_rgb'][off])
   data=dict(label=label,score=value['score'],subscores=scores,
             confidence={a['name']:a['confidence'] for a in receipt['official_response']['answers']})
   if label.startswith('candidate_'):
    pred=preds[int(label[-1])]
    values=metric(pred.astype(np.float32)/255,truth,[32])
    data['metrics_all32']=values['metrics']['32']
    offsets=packet['sampled_prediction_offsets']
    data['metrics_sampled8']={k:float(np.mean(np.array(v)[offsets])) for k,v in values['per_frame'].items()}
    data['per_frame']=values['per_frame']
    case['candidates'].append(data)
   else:case['controls'][label]=data
  result.append(case)
  if group['id'].startswith('g00045'):
   canvas=Image.new('RGB',(960,4*280),'white');draw=ImageDraw.Draw(canvas)
   for row,t in enumerate([8,17,26,31]):
    for col,frames in enumerate([truth,preds[0],preds[1]]):
     canvas.paste(Image.fromarray(frames[t]),(col*320,row*280+24))
     draw.text((col*320+3,row*280+3),f'{"GT" if col==0 else "C"+str(col-1)} frame={t+2}',fill='black')
   canvas.save(out/'wrong_pair_fullsize.jpg')
   with np.load(exp/'reviews'/group['scores']['black']['key']/'frames.npz') as a:
    Image.fromarray(np.concatenate([a['reference_rgb'][-1],a['prediction_rgb'][-1]],axis=1)).save(out/'actual_black_pair.png')
(out/'offline_audit.json').write_text(json.dumps(dict(passed_integrity=True,groups=result),indent=2))
for c in result:
 print(c['id'])
 for v in c['candidates']:print(v['label'],round(v['score'],4),{k:round(x,3) for k,x in v['subscores'].items()},'LPIPSall/sample',round(v['metrics_all32']['lpips'],5),round(v['metrics_sampled8']['lpips'],5))
 print('black',c['controls']['black']['subscores'])

