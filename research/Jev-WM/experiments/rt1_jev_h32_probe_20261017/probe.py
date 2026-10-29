from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/src'))
from runtime import api_environment
from futureworlds.pipeline import WorldPipeline
from futureworlds.data import read_manifest,read_case
from futureworlds.search import rollout_group_beam
from futureworlds.rewards import CodecWorker
from futureworlds.batch import TensorBatch
from futureworlds.jev_judge import Store,atomic_json

@torch.no_grad()
def main():
    api_environment();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20,0)
    cfg=json.loads((ROOT/'config.json').read_text());torch.manual_seed(cfg['seed'])
    pipe=WorldPipeline(cfg['bundle'],'rt1','sft','cuda:0',cfg['text_model'])
    pipe.model.eval().requires_grad_(False);worker=CodecWorker(pipe.codec,None)
    rows={r['id']:r for r in read_manifest(cfg['train_manifest'],'rt1')['cases']}
    schedule=json.loads(Path(cfg['schedule']).read_text())[:3]
    results=[]
    for cursor,entry in enumerate(schedule):
        folder=ROOT/f'generated/{cursor:02d}';folder.mkdir(parents=True,exist_ok=True)
        row=rows[entry['id']];images,actions,task=read_case(cfg['train_manifest'],row,pipe.action_dim,32)
        if (folder/'frames.npz').exists():
            with np.load(folder/'frames.npz',allow_pickle=False) as z:pred=z['predictions'].copy()
        else:
            print(json.dumps(dict(stage='generate32',cursor=cursor,id=row['id'])),flush=True)
            ctx,dynamics,acts,pixels,cond=pipe.encode_training_window(images,actions,task)
            frames,_=rollout_group_beam(pipe.model,ctx.reshape(1,-1)+4375,dynamics[1:2],acts,32,penalty=.1,groups=4,beams=2)
            codes=torch.stack([f['response'] for f in frames],1)
            decoded=worker.detokenize(TensorBatch(dict(tokens=codes,ctx_tokens=ctx.expand(4,-1,-1))),None).batch['pixels'][:,1:]
            pred=decoded.clamp(0,1).mul(255).round().to(torch.uint8).permute(0,1,3,4,2).cpu().numpy()
            assert pred.shape==(4,32,256,320,3)
            np.savez_compressed(folder/'frames.npz',initial=images[:2],reference=images[2:34],predictions=pred)
            atomic_json(folder/'metadata.json',dict(id=row['id'],task=task,cursor=cursor))
            del decoded,codes,frames
        result=dict(cursor=cursor,id=row['id'],task=task,horizons={})
        for horizon in [10,32]:
            dc={**cfg['decisions'],'review_dir':str(ROOT/f'reviews_h{horizon}')}
            store=Store(dc)
            values,stats=store.score_group(images[:2],images[2:2+horizon],pred[:,:horizon],task,seed=cfg['seed'],cursor=cursor)
            result['horizons'][str(horizon)]=dict(rewards=values.tolist(),stats=stats)
            print(json.dumps(dict(cursor=cursor,horizon=horizon,rewards=values.tolist(),stats=stats)),flush=True)
        results.append(result);atomic_json(ROOT/'results.json',dict(optimizer_updates=0,results=results))
    summary={str(h):dict(groups=3,pairs=6,accepted_pairs=sum(x['horizons'][str(h)]['stats']['accepted_pairs'] for x in results),
                       informative_groups=sum(any(v!=0 for v in x['horizons'][str(h)]['rewards']) for x in results),
                       tied_answers=sum(x['horizons'][str(h)]['stats']['tied_answers'] for x in results),
                       insufficient_answers=sum(x['horizons'][str(h)]['stats']['insufficient_answers'] for x in results),
                       order_conflicts=sum(x['horizons'][str(h)]['stats']['order_conflicts'] for x in results)) for h in [10,32]}
    atomic_json(ROOT/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
