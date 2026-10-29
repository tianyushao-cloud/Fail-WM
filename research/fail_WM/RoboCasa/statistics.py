import argparse,json
from pathlib import Path
import h5py,numpy as np
from PIL import Image,ImageDraw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/preview5');a=ap.parse_args();root=Path(a.data);episodes=sorted(p for p in root.iterdir() if (p/'metadata.json').exists());rows=[];summ=[]
    for p in episodes:
        m=json.loads((p/'metadata.json').read_text());summ.append(m)
        with h5py.File(p/'trajectory.h5') as f:
            n=len(f['observations/rgb']);indices=np.linspace(0,n-1,5,dtype=int);row=Image.new('RGB',(1600,264),'#202329');draw=ImageDraw.Draw(row)
            draw.text((8,6),f"{p.name} | success={m['success']} | actions={m['actions']} | {m['failure_type']}",fill='white')
            for j,idx in enumerate(indices):
                im=Image.fromarray(f['observations/rgb'][idx]).resize((320,240));row.paste(im,(j*320,24))
            rows.append(row)
    board=Image.new('RGB',(1600,264*len(rows)))
    for i,row in enumerate(rows):board.paste(row,(0,i*264))
    board.save(root/'contact_sheet.jpg',quality=90)
    stats={'episodes':len(summ),'successes':sum(m['success'] for m in summ),'success_rate':np.mean([m['success'] for m in summ]),'action_lengths':[m['actions'] for m in summ],
       'initial_state_groups':len({m['initial_state_id'] for m in summ}),'initial_state_hashes':len({m['initial_state_sha256'] for m in summ}),'split':'all 5 branches are train; no val/test for this one-group preview'}
    (root/'statistics.json').write_text(json.dumps(stats,indent=2));(root/'summary.json').write_text(json.dumps(summ,indent=2));print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
