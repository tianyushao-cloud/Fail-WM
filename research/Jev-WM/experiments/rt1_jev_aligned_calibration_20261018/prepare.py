from pathlib import Path
import json,sys,shutil
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
EXP=ROOT.parent
OLD=EXP/'rt1_jev_segment_calibration_20261018'
SOURCE=EXP/'rt1_jev_h32_probe_20261017/source/src'
sys.path.insert(0,str(SOURCE))
from futureworlds.pipeline import WorldPipeline,encode_actions
from futureworlds.search import rollout_group_beam
from futureworlds.data import read_manifest,read_case
from futureworlds.batch import TensorBatch
from futureworlds.rewards import CodecWorker
from futureworlds.jev_judge import atomic_json,sha

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20,0);torch.manual_seed(20260915)
    cases=[]
    for meta in json.loads((OLD/'cases.json').read_text()):
        if meta['kind'] not in ['sft_recap','rl_recap'] and meta['id']!='scene3_sft_full':continue
        folder=ROOT/'cases'/meta['id'];folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(OLD/'cases'/meta['id']/'frames.npz',folder/'frames.npz')
        meta={**meta,'cohort':'old_train_diagnostic'};atomic_json(folder/'metadata.json',meta);cases.append(meta)
    manifest=EXP/'rt1_memspo200_upstream_20261017/data/validation/manifest.json'
    rows=read_manifest(str(manifest),'rt1')['cases'][:2]
    train=read_manifest(str(EXP/'rt1_memspo200_upstream_20261017/data/train/manifest.json'),'rt1')['cases']
    assert not set(x['episode_id'] for x in rows)&set(x['episode_id'] for x in train)
    atomic_json(ROOT/'selection.json',dict(rule='First two validation manifest entries in existing order, selected before inference/judging; no cherry-picking',cases=rows,used_for_judge_development=True,not_final_test=True))
    pipe=WorldPipeline(str(EXP/'rt1_memspo200_upstream_20261017/rl_bundle'),'rt1','main','cuda:0','/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base')
    pipe.model.eval().requires_grad_(False);worker=CodecWorker(pipe.codec,None)
    for n,row in enumerate(rows,1):
        initial_folder=ROOT/'generated'/f'val{n}';initial_folder.mkdir(parents=True,exist_ok=True)
        images,actions,task=read_case(str(manifest),row,pipe.action_dim,32)
        if not (initial_folder/'frames.npz').exists():
            print(json.dumps(dict(stage='generate',case=f'val{n}',task=task)),flush=True)
            ctx,anchor=pipe.encode_observations(images[:2]);acts=encode_actions(torch.as_tensor(actions,device=pipe.device),pipe.ranges,pipe.action_dim);pipe.condition(task)
            frames,trace=rollout_group_beam(pipe.model,ctx.reshape(1,-1)+4375,anchor,acts,32,penalty=.1,groups=4,beams=2)
            codes=torch.stack([f['response'] for f in frames],1)
            pix=worker.detokenize(TensorBatch(dict(tokens=codes,ctx_tokens=ctx.expand(4,-1,-1))),None).batch['pixels'][:,1:]
            preds=pix.clamp(0,1).mul(255).round().to(torch.uint8).permute(0,1,3,4,2).cpu().numpy()
            np.savez_compressed(initial_folder/'frames.npz',initial=images[:2],reference=images[2:34],predictions=preds,codes=codes.cpu().numpy())
            atomic_json(initial_folder/'trace.json',trace);del pix,codes,frames
        with np.load(initial_folder/'frames.npz') as z:preds=z['predictions'].copy()
        ref=images[2:34];frozen=ref.copy();frozen[16:]=ref[15];rev=ref.copy();rev[16:]=ref[16:][::-1]
        specs=[(f'rl_g{g}',preds[g],ref,g) for g in range(4)]+[('gt',ref,ref,None),('freeze',frozen,ref,None),('reverse_late',rev,ref,None),('matched_pause',frozen,frozen,None)]
        for kind,pred,reference,group in specs:
            key=f'val{n}_{kind}';folder=ROOT/'cases'/key;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'frames.npz',initial=images[:2],reference=reference,prediction=pred)
            meta=dict(id=key,scene=n+3,kind=kind,group=group,task=task,episode=row['episode_id'],cohort='validation_judge_development',synthetic=kind in ['freeze','reverse_late','matched_pause'],arrays={k:sha(v.tobytes()) for k,v in [('initial',images[:2]),('reference',reference),('prediction',pred)]})
            atomic_json(folder/'metadata.json',meta);cases.append(meta)
        print(json.dumps(dict(stage='prepared',case=f'val{n}')),flush=True)
    atomic_json(ROOT/'cases.json',cases);print('DONE',len(cases),flush=True)

if __name__=='__main__':main()
