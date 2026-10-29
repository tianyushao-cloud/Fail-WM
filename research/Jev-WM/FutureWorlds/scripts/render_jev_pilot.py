from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
root=Path('/media/tyshao/Jev-WM/FutureWorlds/runs/jev_pilot_20261012')
r=json.loads((root/'comparison.json').read_text())
labels=['SFT','R0: 2 steps','Hybrid: 2 steps','R0: 20 steps','Frozen frame']
arms=['sft','base_2step','hybrid_2step','base','frozen']
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
for ax,h in zip(axes,['4','8']):
 values=[r['aggregate'][a][h]['lpips'] for a in arms]
 bars=ax.bar(range(5),values,color=['#64748b','#2563eb','#16a34a','#7c3aed','#ef4444'])
 ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
 ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.set_title(f'{h}-frame LPIPS (lower is better)')
 ax.set_ylim(0,max(values)*1.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Exploratory pilot: 8 held-out states, one seed',fontweight='bold')
fig.savefig(root/'comparison.png',dpi=160)
# First four locked cases, not selected for favorable results.
for case in range(4):
 with np.load(root/'evaluation/sft'/f'case_{case:06d}'/'prediction.npz') as z:
  truth=z['truth'];initial=z['observations'][1]
 rows=[('Ground truth',truth)]
 for a,label in zip(arms[:-1],labels[:-1]):
  with np.load(root/'evaluation'/a/f'case_{case:06d}'/'prediction.npz') as z:
   pred=(z['prediction'].clip(0,1)*255).round().astype(np.uint8)
  rows.append((label,pred))
 rows.append(('Frozen frame',np.repeat(initial[None],8,axis=0)))
 canvas=Image.new('RGB',(8*160,6*152),'white');d=ImageDraw.Draw(canvas)
 for y,(label,frames) in enumerate(rows):
  d.text((4,y*152+3),label,fill='black')
  for x,frame in enumerate(frames):canvas.paste(Image.fromarray(frame).resize((160,128)),(x*160,y*152+22))
 canvas.save(root/f'qualitative_{case:02d}.jpg')
print(root/'comparison.png')
