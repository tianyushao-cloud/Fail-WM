import json,time,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from futureworlds.pipeline import WorldPipeline
from futureworlds.data import read_case
r=Path('/media/tyshao/Jev-WM/FutureWorlds');run=r/'runs/local_success_memspo'
out=run/'smoke';out.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
pipe=WorldPipeline(r/'assets/local_success_sft','robocasa','sft','cuda','/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base')
m=run/'data/validation/manifest.json'
row=json.loads(m.read_text())['cases'][0]
images,actions,text=read_case(m,row,52,10)
t=time.monotonic()
with torch.inference_mode():
    c,anchor=pipe.encode_observations(images[:2])
    ct,d,a,px,cond=pipe.encode_training_window(images,actions,text)
    assert torch.equal(c,ct) and torch.equal(anchor,d[1:2])
    result=pipe.predict(images[:2],actions,text,10,4,'full')
assert np.isfinite(result['prediction']).all()
assert result['prediction'].shape==(10,256,320,3)
np.savez_compressed(out/'first_two_prediction.npz',prediction=result['prediction'],tokens=result['tokens'],truth=images[2:])
movie=np.rint(result['prediction']*255).clip(0,255).astype(np.uint8)
sheet=Image.new('RGB',(640,280*3),'black');draw=ImageDraw.Draw(sheet)
for i,k in enumerate([0,4,9]):
    sheet.paste(Image.fromarray(images[k+2]),(0,280*i+24))
    sheet.paste(Image.fromarray(movie[k]),(320,280*i+24))
    draw.text((4,280*i+4),f'GT absolute t={k+2}',fill='white')
    draw.text((324,280*i+4),'SFT original first two',fill='white')
sheet.save(out/'first_two_keyframes.jpg',quality=90)
receipt=dict(state='PASS',case=row['id'],source_start=row['source_start'],initial_observations=2,horizon=10,training_and_inference_context_equal=True,training_and_inference_anchor_equal=True,finite_prediction=True,beam_width=4,memory='full',precision='FP32',seconds=time.monotonic()-t,formal_rl_started=False)
(out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

