"""torchrun assigns one event-evaluation shard to each visible CUDA device."""
import os
import runpy
import sys
from pathlib import Path
root=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
out=root/'evals_2x2_20261012'
snapshot=root/'experiments_e2v2_adrop_20261012/gt_handoff_frozen.py'
sys.argv=[str(snapshot),'run','--out',str(out),'--labels','start','+2',
          '--shard',os.environ['RANK'],'--shards',os.environ['WORLD_SIZE'],
          '--device','cuda:'+os.environ['LOCAL_RANK']]
for label in ('B','C','D'):sys.argv+=['--model',label+'='+str(out/'bundles'/label)]
runpy.run_path(str(snapshot),run_name='__main__')
