from pathlib import Path
import json,gc,time
import torch
from futureworlds.training import train
from futureworlds.exporting import export_model
from futureworlds.evaluation import evaluate
from futureworlds.data import write_json
ROOT=Path('/media/tyshao/Jev-WM/FutureWorlds/runs/jev_pilot_20261012')
torch.cuda.set_per_process_memory_fraction(.25,0)
for arm in ['base','hybrid']:
 name=arm+'_2step';cfg=json.loads((ROOT/(name+'.json')).read_text())
 torch.cuda.reset_peak_memory_stats();started=time.monotonic()
 train(ROOT/(name+'.json'))
 write_json(ROOT/name/'resources.json',dict(seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
 gc.collect();torch.cuda.empty_cache()
 export_model(cfg['bundle'],'robocasa',ROOT/name/'checkpoints/step_000002',ROOT/'bundles'/name)
 evaluate(str(ROOT/'bundles'/name),'robocasa','main',cfg['evaluation_manifest'],str(ROOT/'evaluation'/name),device='cuda',metric_device='cuda',text_model=cfg['text_model'],horizons=[4,8],beam_width=4)
 gc.collect();torch.cuda.empty_cache()
