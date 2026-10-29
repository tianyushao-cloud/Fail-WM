import json,subprocess
from pathlib import Path
import numpy as np
from futureworlds.robocasa_object_reward import rule_scores
from futureworlds.checkpoints import verify_bundle,file_sha256
R=Path(__file__).parent;D=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
cfg=json.loads((R/'train.json').read_text());m=verify_bundle(cfg['bundle'])
ck=D/'runs/natural_sft_mixed_codec34000_10k_20261011/checkpoints/step_010000'
assert m['training_checkpoint_sha256']==file_sha256(ck/'experiment.json')
# Behavioral checks: same images, vanished cube, duplicate cube, and known false lift.
rows={r['id']:r for r in json.loads((D/'robocasa_data/val.json').read_text())['cases']}
row=rows['g00014_b06']
with np.load(D/'robocasa_data'/row['file']) as z:ref=z['images'][2:34].copy()
p=D/'previews_wm_codec34k_20261012/g00014_b06/natural.mp4'
raw=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout
bad=np.frombuffer(raw,np.uint8).reshape(-1,256,320,3)[:32]
gone=ref.copy();x=gone.astype(float);mask=(x[...,0]>75)&(x[...,0]>1.5*x[...,1])&(x[...,0]>1.4*x[...,2]);gone[mask]=128
dup=ref.copy();dup[:,70:80,200:210]=[230,0,0]
results={k:rule_scores(ref,v)[0] for k,v in dict(exact=ref,false_lift=bad,vanished=gone,duplicated=dup).items()}
assert results['exact']['position']==1 and results['exact']['persistence']==1
assert results['false_lift']['position']<.8
assert results['vanished']['position']<.1 and results['vanished']['persistence']<.1
assert results['duplicated']['persistence']<.5
# Freeze a real successful trajectory; movement matching must score it worse.
r=next(r for r in rows.values() if r['success'] and r['frames']>=34)
with np.load(D/'robocasa_data'/r['file']) as z:succ=z['images'][2:34].copy()
results['frozen_success']=rule_scores(succ,np.repeat(succ[:1],len(succ),axis=0))[0]
assert results['frozen_success']['position']<.9
(R/'checks/reward_rules.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
