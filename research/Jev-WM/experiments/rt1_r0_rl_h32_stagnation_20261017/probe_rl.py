from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
BASE=Path('/media/tyshao/Jev-WM/experiments/rt1_jev_h32_probe_20261017')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source/src'))
from futureworlds.pipeline import WorldPipeline
from futureworlds.data import read_manifest,read_case
from futureworlds.search import rollout_group_beam
from futureworlds.rewards import CodecWorker
from futureworlds.batch import TensorBatch
from futureworlds.jev_judge import atomic_json

def stats(clip):
    hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in clip]
    repeated=np.array([a==b for a,b in zip(hashes,hashes[1:])])
    starts=[0]+(np.flatnonzero(~repeated)+1).tolist()
    runs=[dict(first=s+1,last=e,length=e-s) for s,e in zip(starts,starts[1:]+[len(clip)])]
    return dict(unique_frames=len(set(hashes)),adjacent_repeats=int(repeated.sum()),
                longest_constant_run=max(runs,key=lambda x:x['length']),
                repeated_future_frames=(np.flatnonzero(repeated)+2).tolist())

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20,0)
    cfg=json.loads((BASE/'config.json').read_text());torch.manual_seed(cfg['seed'])
    bundle=BASE.parent/'rt1_memspo200_upstream_20261017/rl_bundle'
    pipe=WorldPipeline(str(bundle),'rt1','main','cuda:0',cfg['text_model'])
    pipe.model.eval().requires_grad_(False);worker=CodecWorker(pipe.codec,None)
    rows={r['id']:r for r in read_manifest(cfg['train_manifest'],'rt1')['cases']}
    schedule=json.loads(Path(cfg['schedule']).read_text())[:3]
    atomic_json(ROOT/'protocol.json',dict(baseline=str(BASE),bundle=str(bundle),variant='main',
       checkpoint='R0 MemSPO step200, trained horizon10',generation_horizon=32,groups=4,beams=2,diversity_penalty=.1,
       seed=cfg['seed'],ids=[x['id'] for x in schedule],updates=0,judge_api_calls=0,
       caveat='These three windows belong to RL training data. Exploratory same-input diagnosis, not held-out generalization. Candidate indices identify beam groups, not paired identical futures.'))
    results=[]
    for cursor,entry in enumerate(schedule):
        folder=ROOT/f'generated/{cursor:02d}';folder.mkdir(parents=True,exist_ok=True)
        row=rows[entry['id']];images,actions,task=read_case(cfg['train_manifest'],row,pipe.action_dim,32)
        with np.load(BASE/f'generated/{cursor:02d}/frames.npz') as z:
            sft=z['predictions'].copy();assert np.array_equal(images[:2],z['initial']) and np.array_equal(images[2:34],z['reference'])
        if (folder/'frames.npz').exists():
            with np.load(folder/'frames.npz') as z:pred=z['predictions'].copy();codes_np=z['codes'].copy()
        else:
            print(json.dumps(dict(stage='generate_rl',scene=cursor+1,id=row['id'])),flush=True)
            ctx,dynamics,acts,pixels,cond=pipe.encode_training_window(images,actions,task)
            frames,trace=rollout_group_beam(pipe.model,ctx.reshape(1,-1)+4375,dynamics[1:2],acts,32,penalty=.1,groups=4,beams=2)
            codes=torch.stack([f['response'] for f in frames],1);codes_np=codes.cpu().numpy()
            decoded=worker.detokenize(TensorBatch(dict(tokens=codes,ctx_tokens=ctx.expand(4,-1,-1))),None).batch['pixels'][:,1:]
            pred=decoded.clamp(0,1).mul(255).round().to(torch.uint8).permute(0,1,3,4,2).cpu().numpy()
            assert pred.shape==(4,32,256,320,3)
            np.savez_compressed(folder/'frames.npz',initial=images[:2],reference=images[2:34],predictions=pred,codes=codes_np)
            atomic_json(folder/'metadata.json',dict(id=row['id'],task=task,cursor=cursor))
            atomic_json(folder/'search_trace.json',trace)
            del decoded,codes,frames
        gt=images[2:34]
        result=dict(scene=cursor+1,id=row['id'],task=task,GT=stats(gt),SFT=[stats(x) for x in sft],RL=[stats(x) for x in pred])
        for i in range(4):
            same_tokens=np.all(codes_np[i,1:]==codes_np[i,:-1],axis=-1)
            result['RL'][i]['repeated_tokens_match_rgb']=bool(np.array_equal(same_tokens,np.all(pred[i,1:]==pred[i,:-1],axis=(1,2,3))))
        # Pixel differences are supporting offline diagnostics, not judge rewards.
        for name,clips in [('SFT',sft),('RL',pred)]:
            for item,clip in zip(result[name],clips):item['mean_mae_to_gt']=float(np.abs(clip.astype(np.float32)-gt.astype(np.float32)).mean()/255)
        results.append(result);atomic_json(ROOT/'results.json',dict(results=results,optimizer_updates=0))
        print(json.dumps(dict(scene=cursor+1,SFT_repeats=[x['adjacent_repeats'] for x in result['SFT']],RL_repeats=[x['adjacent_repeats'] for x in result['RL']],SFT_longest=[x['longest_constant_run']['length'] for x in result['SFT']],RL_longest=[x['longest_constant_run']['length'] for x in result['RL']])),flush=True)
    totals={name:dict(adjacent_repeats=sum(x['adjacent_repeats'] for r in results for x in r[name]),
                     transitions=3*4*31,clips_with_run_ge10=sum(x['longest_constant_run']['length']>=10 for r in results for x in r[name])) for name in ['SFT','RL']}
    atomic_json(ROOT/'summary.json',totals);print(json.dumps(totals),flush=True)

if __name__=='__main__':main()
