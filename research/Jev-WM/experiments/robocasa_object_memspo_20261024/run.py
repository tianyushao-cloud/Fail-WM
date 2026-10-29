import os,json,time,traceback,fcntl
from pathlib import Path
R=Path(__file__).parent
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import sys
sys.path.insert(0,str(R/'source/src'))
from futureworlds.data import write_json
lock=(R/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
import torch
torch.cuda.set_per_process_memory_fraction(42*1024**3/torch.cuda.get_device_properties(0).total_memory)
write_json(R/'status.json',dict(state='RUNNING',pid=os.getpid(),started=time.time(),steps=3))
try:
 from futureworlds.training import train
 train(R/'train.json')
 write_json(R/'status.json',dict(state='COMPLETE',steps=3,finished=time.time(),peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3))
except Exception as e:
 write_json(R/'status.json',dict(state='FAILED',error=repr(e),time=time.time()))
 traceback.print_exc();raise
