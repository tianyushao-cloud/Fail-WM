import os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0';os.environ['OMP_NUM_THREADS']='2'
import json,time
from pathlib import Path
import torch
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(.28,0)
import futureworlds.pipeline as pipeline
original=pipeline.beam_decode
counter=0
def logged(*args,**kwargs):
 global counter
 out=original(*args,**kwargs);counter+=1
 print(json.dumps(dict(event='frame_generated',frame=counter,total=64,time=time.time())),flush=True)
 return out
pipeline.beam_decode=logged
from futureworlds.evaluation import evaluate
root=Path('/media/tyshao/Jev-WM/FutureWorlds/runs/short_success_sft_preview')
evaluate(bundle='/media/tyshao/Jev-WM/FutureWorlds/assets/local_success_sft',dataset='robocasa',variant='sft',
 manifest=str(root/'manifest.json'),output=str(root/'evaluation'),device='cuda',metric_device='cuda',
 text_model='/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base',horizons=[8,16,32],beam_width=4,memory='full')
print('SFT_INFERENCE_COMPLETE',flush=True)

