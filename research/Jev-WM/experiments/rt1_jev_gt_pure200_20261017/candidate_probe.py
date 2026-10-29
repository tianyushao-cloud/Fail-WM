"""Small inference-only SFT candidate probe; zero optimizer steps."""
import json
import os
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/src'))
from futureworlds.pipeline import WorldPipeline
from futureworlds.data import read_manifest,read_case
from futureworlds.search import rollout_group_beam
from futureworlds.jev_gt_reward import JevGTReward
from futureworlds.jev_judge import atomic_json
from runtime import api_environment


@torch.no_grad()
def main():
    api_environment();torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20,0)
    cfg=json.loads((ROOT/'train.json').read_text())
    torch.manual_seed(cfg['seed'])
    dc={**cfg['decisions'],'backend':'api','review_dir':str(ROOT/'candidate_probe_reviews'),'max_http_attempts':24}
    pipe=WorldPipeline(cfg['bundle'],'rt1','sft','cuda:0',cfg['text_model'])
    pipe.model.eval().requires_grad_(False)
    reward=JevGTReward(pipe.codec,dc)
    rows={r['id']:r for r in read_manifest(cfg['train_manifest'],'rt1')['cases']}
    schedule=json.loads(Path(cfg['schedule']).read_text())
    results=[]
    for cursor,entry in enumerate(schedule[:3]):
        row=rows[entry['id']];images,actions,text=read_case(cfg['train_manifest'],row,pipe.action_dim,10)
        ctx,dynamics,acts,pixels,cond=pipe.encode_training_window(images,actions,text)
        frames,_=rollout_group_beam(pipe.model,ctx.reshape(1,-1)+4375,dynamics[1:2],acts,10,
                                    penalty=.1,groups=4,beams=2)
        values,stats=reward(frames,ctx,acts,pixels,task=text,cursor=cursor,seed=cfg['seed'])
        result=dict(cursor=cursor,id=row['id'],rewards=values.cpu().tolist(),stats=stats)
        results.append(result);print(json.dumps(result),flush=True)
        atomic_json(ROOT/'candidate_probe.json',dict(optimizer_updates=0,results=results,
            note='First three scheduled training windows, inference-only; no hyperparameter selection based on evaluation data.'))
    print(json.dumps(dict(informative_groups=sum(any(v!=0 for v in r['rewards']) for r in results),groups=len(results))),flush=True)


if __name__=='__main__':main()
