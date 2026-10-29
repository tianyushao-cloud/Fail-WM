from pathlib import Path
import json,sys
import numpy as np
import torch
ROOT=Path('/media/tyshao/Jev-WM/experiments/rt1_jev_h32_probe_20261017')
sys.path.insert(0,str(ROOT/'source/src'))
from futureworlds.pipeline import WorldPipeline,encode_actions
from futureworlds.data import read_case,read_manifest
from futureworlds.search import rollout_group_beam
from futureworlds.rewards import CodecWorker
from futureworlds.batch import TensorBatch
from futureworlds.jev_judge import atomic_json

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20,0)
    cfg=json.loads((ROOT/'config.json').read_text());torch.manual_seed(cfg['seed'])
    pipe=WorldPipeline(cfg['bundle'],'rt1','sft','cuda:0',cfg['text_model']);pipe.model.eval().requires_grad_(False)
    rows={r['id']:r for r in read_manifest(cfg['train_manifest'],'rt1')['cases']}
    meta=json.loads((ROOT/'generated/01/metadata.json').read_text())
    images,actions,task=read_case(cfg['train_manifest'],rows[meta['id']],13,32)
    ctx,dyn,acts,pixels,cond=pipe.encode_training_window(images,actions,task)
    initial_ctx,initial_anchor=pipe.encode_observations(images[:2])
    assert torch.equal(ctx,initial_ctx) and torch.equal(dyn[1:2],initial_anchor)
    frames,trace=rollout_group_beam(pipe.model,ctx.reshape(1,-1)+4375,dyn[1:2],acts,32,penalty=.1,groups=4,beams=2)
    codes=torch.stack([f['response'] for f in frames],1)
    pred=CodecWorker(pipe.codec,None).detokenize(TensorBatch(dict(tokens=codes,ctx_tokens=ctx.expand(4,-1,-1))),None).batch['pixels'][:,1:]
    rgb=pred.clamp(0,1).mul(255).round().to(torch.uint8).permute(0,1,3,4,2).cpu().numpy()
    with np.load(ROOT/'generated/01/frames.npz') as z:old=z['predictions'].copy()
    info=dict(scene=2,rerun_matches_saved_rgb=bool(np.array_equal(old,rgb)),initial_encoding_matches_observation_only=True,
              unique_encoded_action_rows=int(torch.unique(acts[1:33],dim=0).shape[0]),candidates=[])
    for i in range(4):
        repeated=(codes[i,1:]==codes[i,:-1]).all(-1).cpu().numpy()
        rgb_repeated=np.all(rgb[i,1:]==rgb[i,:-1],axis=(1,2,3))
        # one-indexed future-frame numbers where frame equals its predecessor
        info['candidates'].append(dict(candidate=i,unique_token_frames=int(torch.unique(codes[i],dim=0).shape[0]),
            repeated_future_frames=(np.flatnonzero(repeated)+2).tolist(),
            token_repeats_match_rgb=bool(np.array_equal(repeated,rgb_repeated))))
    raw=torch.as_tensor(actions,device=pipe.device)
    variable=pipe.ranges[:,1]>pipe.ranges[:,0]
    out=((raw[1:33]<pipe.ranges[:,0])|(raw[1:33]>pipe.ranges[:,1]))[:,variable]
    info['out_of_range_variable_action_values']=int(out.sum());info['variable_action_values']=out.numel()
    np.savez_compressed(ROOT/'scene2_token_diagnostics.npz',codes=codes.cpu().numpy(),encoded_actions=acts.cpu().numpy())
    atomic_json(ROOT/'scene2_token_diagnostics.json',info);atomic_json(ROOT/'scene2_search_trace.json',trace)
    print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
