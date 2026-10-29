from pathlib import Path
import sys, json, hashlib, time, gc
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'rt1_jev_h32_probe_20261017'
RLBASE = ROOT.parent / 'rt1_r0_rl_h32_stagnation_20261017'
sys.path.insert(0, str(BASE / 'source/src'))
from futureworlds.pipeline import WorldPipeline
from futureworlds.data import read_manifest, read_case
import futureworlds.search as search
from futureworlds.rewards import CodecWorker
from futureworlds.batch import TensorBatch
from futureworlds.jev_judge import atomic_json

original_prefix = search.make_prefix
def full_prefix(context, history, actions, target):
    kept = list(range(1, target))
    blocks = [context]
    for t in kept:
        blocks += [history[t], actions[t].expand(context.shape[0], -1)]
    prefix = torch.cat(blocks, dim=1)
    if target <= 8:
        expected, indices = original_prefix(context, history, actions, target)
        assert kept == indices and torch.equal(prefix, expected)
    return prefix, kept

def stats(clip, gt):
    repeated = np.all(clip[1:] == clip[:-1], axis=(1,2,3))
    starts = [0] + (np.flatnonzero(~repeated)+1).tolist()
    return dict(adjacent_repeats=int(repeated.sum()), longest_constant_run=max(e-s for s,e in zip(starts, starts[1:]+[len(clip)])), mean_mae_to_gt=float(np.abs(clip.astype(np.float32)-gt.astype(np.float32)).mean()/255))

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    cfg = json.loads((BASE/'config.json').read_text())
    torch.manual_seed(cfg['seed'])
    rows = {r['id']:r for r in read_manifest(cfg['train_manifest'],'rt1')['cases']}
    schedule = json.loads(Path(cfg['schedule']).read_text())[:3]
    protocol = dict(horizon=32, initial_observations=2, groups=4, beams_per_group=2, diversity_penalty=.1, seed=cfg['seed'], full_history='Scene context plus every complete state-action block, with the same per-frame KV/position reset as ReCAP.', recap='Scene context plus dynamic anchor1 and recent6, reused from the previous verified experiment.', comparison='Only context retention changes within each checkpoint; no future GT refresh.', weights=dict(SFT=cfg['bundle'],RL=str(BASE.parent/'rt1_memspo200_upstream_20261017/rl_bundle')), caveat='Three training windows: diagnostic, not held-out generalization. Beam-group indices are not identical future trajectories. All four groups included.', updates=0, judge_api_calls=0)
    atomic_json(ROOT/'protocol.json',protocol)
    search.make_prefix = full_prefix
    results = []
    for model in ['SFT','RL']:
        bundle = cfg['bundle'] if model=='SFT' else str(BASE.parent/'rt1_memspo200_upstream_20261017/rl_bundle')
        pipe = WorldPipeline(bundle,'rt1','sft' if model=='SFT' else 'main','cuda:0',cfg['text_model'])
        pipe.model.eval().requires_grad_(False)
        worker = CodecWorker(pipe.codec,None)
        for cursor,entry in enumerate(schedule):
            folder = ROOT/f'generated/{model}/{cursor:02d}'
            folder.mkdir(parents=True,exist_ok=True)
            images,actions,task = read_case(cfg['train_manifest'],rows[entry['id']],pipe.action_dim,32)
            baseline = (BASE if model=='SFT' else RLBASE)/f'generated/{cursor:02d}/frames.npz'
            with np.load(baseline) as z:
                recap = z['predictions'].copy()
                assert np.array_equal(images[:2],z['initial']) and np.array_equal(images[2:34],z['reference'])
            if not (folder/'frames.npz').exists():
                start = time.monotonic()
                print(json.dumps(dict(stage='generate',model=model,scene=cursor+1)),flush=True)
                ctx,anchor = pipe.encode_observations(images[:2])
                from futureworlds.pipeline import encode_actions
                acts = encode_actions(torch.as_tensor(actions,device=pipe.device),pipe.ranges,pipe.action_dim)
                pipe.condition(task)
                frames,trace = search.rollout_group_beam(pipe.model,ctx.reshape(1,-1)+4375,anchor,acts,32,penalty=.1,groups=4,beams=2)
                assert all(f['retained']==list(range(1,f['target'])) for f in trace['frames'])
                codes = torch.stack([f['response'] for f in frames],1)
                decoded = worker.detokenize(TensorBatch(dict(tokens=codes,ctx_tokens=ctx.expand(4,-1,-1))),None).batch['pixels'][:,1:]
                pred = decoded.clamp(0,1).mul(255).round().to(torch.uint8).permute(0,1,3,4,2).cpu().numpy()
                assert pred.shape==(4,32,256,320,3)
                np.savez_compressed(folder/'frames.npz',initial=images[:2],reference=images[2:34],full=pred,recap=recap,codes=codes.cpu().numpy())
                atomic_json(folder/'search_trace.json',trace)
                del frames,codes,decoded
                print(json.dumps(dict(stage='generated',model=model,scene=cursor+1,seconds=time.monotonic()-start)),flush=True)
            with np.load(folder/'frames.npz') as z:
                pred=z['full'].copy()
            item=dict(model=model,scene=cursor+1,id=entry['id'],task=task,full=[stats(x,images[2:34]) for x in pred],recap=[stats(x,images[2:34]) for x in recap])
            results.append(item)
            atomic_json(ROOT/'results.json',results)
            print(json.dumps(item),flush=True)
        del worker,pipe
        gc.collect();torch.cuda.empty_cache()
    print('DONE',flush=True)

if __name__=='__main__':main()
