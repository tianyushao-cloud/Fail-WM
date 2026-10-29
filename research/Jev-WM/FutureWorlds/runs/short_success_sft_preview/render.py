import json
from pathlib import Path
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent
summary=json.loads((R/'evaluation/summary.json').read_text())
assert summary['state']=='COMPLETE' and len(summary['per_case'])==2
rows=json.loads((R/'manifest.json').read_text())['cases']
html=['<!doctype html><meta charset="utf-8"><title>新数据 SFT 预测预览</title>',
'<h1>现有SFT：新采集训练样本预测</h1>',
'<p>左为真实轨迹，右为SFT自回归预测。前两帧两侧均为真实初始观测；之后32帧无真值刷新。5fps，正常速度。仅使用训练样本。</p>']
results=[]
for i,row in enumerate(rows):
 source=R/'evaluation'/f'case_{i:06d}'
 z=np.load(source/'prediction.npz')
 assert z['prediction'].shape==(32,256,320,3)
 gt=np.concatenate([z['observations'],z['truth']])
 pred=np.concatenate([z['observations'],np.rint(np.clip(z['prediction'],0,1)*255).astype(np.uint8)])
 folder=R/row['id'];folder.mkdir(exist_ok=True)
 frames=[]
 for t,(a,b) in enumerate(zip(gt,pred)):
  im=Image.new('RGB',(640,288),(245,245,245))
  im.paste(Image.fromarray(a),(0,32));im.paste(Image.fromarray(b),(320,32))
  d=ImageDraw.Draw(im)
  d.text((8,9),f'Ground truth | {t/5:.1f}s',fill='black')
  d.text((328,9),'SFT | '+('initial observation' if t<2 else f'predicted {t-1}/32'),fill='black')
  frames.append(np.asarray(im))
 imageio.mimwrite(folder/'comparison.mp4',frames,fps=5,codec='libx264',quality=9,macro_block_size=16,ffmpeg_params=['-threads','1'])
 imageio.mimwrite(folder/'sft.mp4',pred,fps=5,codec='libx264',quality=9,macro_block_size=16,ffmpeg_params=['-threads','1'])
 imageio.mimwrite(folder/'ground_truth.mp4',gt,fps=5,codec='libx264',quality=9,macro_block_size=16,ffmpeg_params=['-threads','1'])
 indices=[1,9,13,16,20,24,28,33]
 sheet=Image.new('RGB',(1280,288*4),'white')
 for j,t in enumerate(indices):sheet.paste(Image.fromarray(frames[t]),((j%2)*640,(j//2)*288))
 sheet.save(folder/'contact_sheet.jpg',quality=93)
 res=summary['per_case'][i];assert res['id']==row['id']
 metrics=res['metrics']
 (folder/'metrics.json').write_text(json.dumps(metrics,indent=2))
 html.append(f'<h2>{row["id"]}</h2><p>LPIPS32: {metrics["32"]["lpips"]:.6f}; SSIM32: {metrics["32"]["ssim"]:.6f}</p><video controls loop width="960" src="{row["id"]}/comparison.mp4"></video><p><a href="{row["id"]}/contact_sheet.jpg">查看关键帧</a></p>')
 results.append(dict(id=row['id'],metrics=metrics,comparison=str(folder/'comparison.mp4')))
(R/'index.html').write_text('\n'.join(html))
(R/'preview_summary.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))

