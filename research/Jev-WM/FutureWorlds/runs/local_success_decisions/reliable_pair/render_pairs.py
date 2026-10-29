import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from futureworlds.focused_alignment import crop_box
root=Path('runs/local_success_decisions/reliable_pair/calibration')
for folder in root.iterdir():
 p=folder/'pair_selection.json'
 if not p.exists():continue
 chosen=json.loads(p.read_text())['selection']['chosen']
 if not chosen:continue
 with np.load(folder/'candidates.npz') as z:
  box=crop_box(z['initial'],z['reference'])
  rows=[('GT',z['reference']),(f"WIN {chosen['winner']}",z['prediction'][chosen['winner']]),(f"LOSE {chosen['loser']}",z['prediction'][chosen['loser']])]
  canvas=Image.new('RGB',(8*200,4*3*180),'white');draw=ImageDraw.Draw(canvas)
  for block in range(4):
   for r,(name,seq) in enumerate(rows):
    for c in range(8):
     t=block*8+c;x=c*200;y=(block*3+r)*180
     im=ImageOps.contain(Image.fromarray(seq[t]).crop(box),(200,158),Image.Resampling.NEAREST)
     canvas.paste(im,(x,y+22));draw.text((x+4,y+4),f'{name} t={t+2}',fill='black')
  canvas.save(folder/'selected_pair_all32.jpg')

