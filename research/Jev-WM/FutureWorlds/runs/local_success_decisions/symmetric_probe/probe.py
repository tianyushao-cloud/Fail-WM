
import base64,io,json,hashlib,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from futureworlds.decision_api import DecisionsClient,normalized_scores
from futureworlds.focused_alignment import crop_box
root=Path('/media/tyshao/Jev-WM/FutureWorlds')
out=root/'runs/local_success_decisions/symmetric_probe';out.mkdir(exist_ok=True)
os.environ['OPENAI_API_KEY']=Path('/root/.config/jev-wm/openai.key').read_text().strip()
for k in ['HTTPS_PROXY','HTTP_PROXY','https_proxy','http_proxy']:os.environ[k]='http://127.0.0.1:17897'
PROMPT=("Measure visual agreement between image A and image B of the SAME robot scene at the SAME timestamp. "
        "This is symmetric: exchanging A and B must not change agreement. Neither image is preferred. "
        "FULL shows the scene; DETAIL is an enlarged crop of the same image, not an extra object. "
        "The red cube is the manipulated object; the green disk is a marker. "
        "Compare only visible geometry, count and state. Do not infer hidden actions or assess task success.")
LEVELS=[dict(label=a,description=b) for a,b in [
 ('different','No useful correspondence, black vs visible scene, or fundamentally different state.'),
 ('major_difference','Clear missing/duplicated cube or large position/state mismatch.'),
 ('moderate_difference','Clear moderate position/orientation or visible state mismatch.'),
 ('minor_difference','Only small position/shape differences; count and visible state agree.'),
 ('same','Same visible geometric/state content, allowing negligible codec or texture differences.')]]
QUESTIONS=[dict(type='score',name='geometry',instructions="Compare only the robot and gripper position/orientation, table and major scene geometry in A FULL vs B FULL. Ignore texture, shadows and the cube for this question.",levels=LEVELS),
           dict(type='score',name='cube',instructions="Compare the visible red cube in A vs B, using DETAIL: number of cubes, positions, attachment to gripper vs support on table, and visibility. A single clear extra or missing visible cube is a major difference. Ignore robot geometry unrelated to the cube.",levels=LEVELS)]
def encode(frame,label):
 canvas=Image.new('RGB',(frame.width,frame.height+28),'white');canvas.paste(frame,(0,28));ImageDraw.Draw(canvas).text((10,8),label,fill='black')
 stream=io.BytesIO();canvas.save(stream,format='PNG')
 return dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode())
def payload(a,b,box,reverse):
 if reverse:a,b=b,a
 content=[dict(type='input_text',text=PROMPT)]
 for label,frame in [('A',a),('B',b)]:
  image=Image.fromarray(frame)
  content.extend([encode(image,label+' FULL'),encode(ImageOps.contain(image.crop(box),(384,384),Image.Resampling.NEAREST),label+' DETAIL')])
 return dict(model='gpt-6-luna',input=[dict(role='user',content=content)],questions=QUESTIONS)
client=DecisionsClient(max_calls=24,timeout=90,retries=0,spacing=3)
source=root/'runs/local_success_decisions/calibration/g00045_b09_s000000_n042/candidates.npz'
with np.load(source) as z:
 controls=np.load(root/'runs/local_success_decisions/focused/development/controls.npz')
 box=crop_box(z['initial'],z['reference'])
 cases=[('same',8,z['reference'][8]),('black',8,controls['black'][8]),
        ('duplicate',8,controls['duplicate'][8]),('missing',8,controls['missing'][8]),
        ('freeze',13,controls['frozen'][13])]
 cases += [(f'candidate_{i}',t,z['prediction'][i,t]) for t in [8,13] for i in range(3)]
 result=[]
 for name,t,pred in cases:
  orders=[]
  for rev in [False,True]:
   pl=payload(z['reference'][t],pred,box,rev)
   digest=hashlib.sha256(json.dumps(pl,sort_keys=True).encode()).hexdigest()
   path=out/(digest+'.json')
   if path.exists():row=json.loads(path.read_text())
   else:
    response,transport=client.score(pl)
    row=dict(name=name,frame=t+2,reverse=rev,payload_sha256=digest,
             scores=normalized_scores(response,QUESTIONS,'gpt-6-luna'),
             official_response=response,transport=transport)
    path.write_text(json.dumps(row,indent=2))
   orders.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['official_response','transport','payload_sha256']}),flush=True)
  result.append(dict(name=name,frame=t+2,orders=orders))
 (out/'results.json').write_text(json.dumps(dict(prompt=PROMPT,questions=QUESTIONS,results=result,new_api_calls=client.calls),indent=2))

