import base64,io,json,hashlib,os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from futureworlds.decision_api import DecisionsClient,normalized_scores
from futureworlds.focused_alignment import crop_box
ROOT=Path('/media/tyshao/Jev-WM/FutureWorlds')
OUT=ROOT/'runs/local_success_decisions/observable_probe'
OUT.mkdir(exist_ok=True)
os.environ['OPENAI_API_KEY']=Path('/root/.config/jev-wm/openai.key').read_text().strip()
for k in ['HTTPS_PROXY','HTTP_PROXY','https_proxy','http_proxy']:os.environ[k]='http://127.0.0.1:17897'
PROMPT=("Inspect ONE robot-scene image. Report only what is visibly present in THIS image. "
"The red cuboid is the manipulated object. The green disk is a flat marker, not a cube. "
"Do not assume a successful grasp or infer hidden objects. Robot fingers can be dark gray. "
"A faint pale reddish stain is not a clearly visible solid red cube. "
"Two separated red cuboids, even if one is near the gripper, count as two. "
"Judge visible shape and spatial relation; do not infer physical forces.")
QTEXT={
'visible_cube':"Is at least one clearly visible solid red cuboid present?",
'extra_cube':"Are two or more spatially separate solid red cuboids visible in this single image? Do not count two faces of one cube as separate cubes.",
'table_cube':"Is a solid red cuboid visibly resting on the white tabletop?",
'gripper_cube':"Is a solid red cuboid visibly touching or between the gripper fingers, above the tabletop? Do not count a cube on the tabletop below the fingers unless it visibly touches them.",
'scene_present':"Is the robot/table scene visible, rather than a black/empty image?"
}
QUESTIONS=[dict(type='score',name=n,instructions=t,levels=[dict(label='no',description='The visible image does not support this statement.'),dict(label='yes',description='The visible image supports this statement.')]) for n,t in QTEXT.items()]
def payload(frame,box,size):
 im=ImageOps.contain(Image.fromarray(frame).crop(box),(size,size),Image.Resampling.NEAREST)
 b=io.BytesIO();im.save(b,format='PNG')
 return dict(model='gpt-6-luna',input=[dict(role='user',content=[dict(type='input_text',text=PROMPT),dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode())])],questions=QUESTIONS)
client=DecisionsClient(max_calls=28,timeout=90,retries=0,spacing=3)
with np.load(ROOT/'runs/local_success_decisions/calibration/g00045_b09_s000000_n042/candidates.npz') as z, np.load(ROOT/'runs/local_success_decisions/focused/development/controls.npz') as c:
 box=crop_box(z['initial'],z['reference'])
 cases=[('reference',8,z['reference'][8]),('black',8,c['black'][8]),('duplicate',8,c['duplicate'][8]),('missing',8,c['missing'][8]),('frozen',13,c['frozen'][13]),('reference',13,z['reference'][13])]
 cases +=[(f'candidate_{i}',t,z['prediction'][i,t]) for t in [8,13] for i in range(3)]
 results=[]
 for name,t,frame in cases:
  for size in [384,640]:
   pl=payload(frame,box,size);key=hashlib.sha256(json.dumps(pl,sort_keys=True).encode()).hexdigest();p=OUT/(key+'.json')
   if p.exists():receipt=json.loads(p.read_text())
   else:
    response,transport=client.score(pl)
    receipt=dict(payload_sha256=key,official_response=response,transport=transport)
    p.write_text(json.dumps(receipt,indent=2))
   row=dict(name=name,frame=t+2,size=size,scores=normalized_scores(receipt['official_response'],QUESTIONS,'gpt-6-luna'),key=key)
   results.append(row);print(json.dumps(row),flush=True)
 (OUT/'results.json').write_text(json.dumps(dict(prompt=PROMPT,questions=QUESTIONS,crop_box=box,results=results,new_api_calls=client.calls),indent=2))

