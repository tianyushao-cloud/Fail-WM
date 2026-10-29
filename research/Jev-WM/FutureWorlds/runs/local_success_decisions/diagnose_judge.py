
import base64,hashlib,io,json,os,time,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from futureworlds.decision_api import DecisionsClient,normalized_scores
from futureworlds.reference_alignment import ReferenceAlignmentStore
root=Path('/media/tyshao/Jev-WM/FutureWorlds/runs/local_success_decisions')
out=root/'diagnosis';out.mkdir(exist_ok=True)
group=json.loads((root/'calibration.json').read_text())['results'][0]
if not os.environ.get('OPENAI_API_KEY'):os.environ['OPENAI_API_KEY']=Path('/root/.config/jev-wm/openai.key').read_text().strip()
for k in ['HTTPS_PROXY','HTTP_PROXY','https_proxy','http_proxy']:os.environ[k]='http://127.0.0.1:17897'
client=DecisionsClient(max_calls=9,timeout=90,retries=0,spacing=6)
def image_part(frame):
 stream=io.BytesIO();Image.fromarray(frame).save(stream,format='PNG')
 return dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode())
cases=[('candidate_0','original',1),('candidate_1','original',1),('candidate_0','original',2),('candidate_1','original',2),
       ('black','original',1),('black','separate',1),('black','swapped',1),
       ('candidate_0','separate',1),('candidate_1','separate',1)]
for label,layout,repeat in cases:
 dest=out/f'{label}_{layout}_{repeat}.json'
 if dest.exists():continue
 key=group['scores'][label]['key'];folder=root/'reviews'/key
 packet=json.loads((folder/'request.json').read_text())
 with np.load(folder/'frames.npz') as arrays:
  payload=ReferenceAlignmentStore.build_payload(packet,arrays)
  if layout=='swapped':
   content=payload['input'][0]['content']
   content[0]['text']=content[0]['text'].replace('LEFT is the recorded REFERENCE, RIGHT is the PREDICTION','RIGHT is the recorded REFERENCE, LEFT is the PREDICTION')
   for n,offset in enumerate(packet['sampled_prediction_offsets']):
    content[1+2*n]['text']=f'Frame {offset+2}. RIGHT reference; LEFT prediction.'
    content[2+2*n]=image_part(np.concatenate([arrays['prediction_rgb'][offset],arrays['reference_rgb'][offset]],axis=1))
  elif layout=='separate':
   prompt=packet['prompt'].replace('Each image is a synchronized pair: LEFT is the recorded REFERENCE, RIGHT is the PREDICTION, at the same numbered time. Pairs are in chronological order.',
    'Images come in synchronized pairs of separate images. For each numbered frame the first image is REFERENCE and the second is PREDICTION. Pairs are chronological. Compare the two images at the same timestamp.')
   content=[dict(type='input_text',text=prompt+'\\nTask: '+json.dumps(packet['task']))]
   for offset in packet['sampled_prediction_offsets']:
    for name,role in [('reference_rgb','REFERENCE'),('prediction_rgb','PREDICTION')]:
     content.extend([dict(type='input_text',text=f'Frame {offset+2}: {role}'),image_part(arrays[name][offset])])
   payload['input'][0]['content']=content
 digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
 response,transport=client.score(payload)
 scores=normalized_scores(response,payload['questions'],payload['model'])
 result=dict(label=label,layout=layout,repeat=repeat,source_key=key,payload_sha256=digest,
             scores=scores,mean=sum(scores.values())/len(scores),official_response=response,transport=transport)
 dest.write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ['official_response','transport','source_key','payload_sha256']}),flush=True)

