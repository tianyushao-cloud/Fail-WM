"""Real cached examples: layout invariants and three GPU optimizer updates."""
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from futureworlds.cache import load_cached
from futureworlds.protocol import feature_fingerprint, implementation
from futureworlds.training import sft_example
from batched_sft_adrop_20261012 import _load_actor, _pad_examples, _collate_condition
from action_dropout_20261012 import apply_action_dropout, action_positions, source_identity

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
EXP=ROOT/'experiments_e2v2_adrop_20261012'

def run(device='cuda',memory_gib=6):
    cfg=json.loads((EXP/'C.json').read_text())
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.cuda.set_device(device)
    total=torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(memory_gib*1024**3/total,device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    before=implementation()
    fh=feature_fingerprint(cfg['bundle'],'robocasa',cfg['text_model'])
    assert fh==json.loads((Path(cfg['token_cache'])/'complete.json').read_text())['bundle_sha256']
    rows=json.loads(Path(cfg['train_manifest']).read_text())['cases']
    chosen=[next(r for r in rows if r['success']),next(r for r in rows if not r['success'])]
    cached=[load_cached(cfg['token_cache'],r,fh,'cpu') for r in chosen]
    checks=[]
    for row,(ctx,dyn,acts,cond) in zip(chosen,cached):
        for target in (2,7,20,len(dyn)-1):
            ex=sft_example(ctx,dyn,acts,target)
            assert not (ex['input_ids']==9006).any()
            pos=action_positions(ex)
            assert ((ex['input_ids'][:,pos]>=8750)&(ex['input_ids'][:,pos]<=9005)).all()
            # Check position selector against independent token-domain mask.
            expected=(ex['input_ids']>=8750)&(ex['input_ids']<=9005)
            mask=torch.zeros_like(expected);mask[:,pos]=True
            assert torch.equal(mask,expected)
            zero,flag=apply_action_dropout(ex,0,cfg['seed'],0)
            assert not flag and all(torch.equal(ex[k],zero[k]) for k in ex)
            null,flag=apply_action_dropout(ex,0,cfg['seed'],1)
            assert flag and (null['input_ids'][mask]==9006).all()
            assert torch.equal(null['input_ids'][~mask],ex['input_ids'][~mask])
            assert all(torch.equal(null[k],ex[k]) for k in ('labels','attention_mask','position_ids'))
            assert not (ex['input_ids']==9006).any() # no mutation of original/cache
            checks.append(dict(id=row['id'],target=target,action_tokens=len(pos)))
    settings=json.loads((Path(cfg['bundle'])/'robocasa/futureworlds_config.json').read_text())
    actor=_load_actor(cfg,SimpleNamespace(model=None,settings=settings),device,'sft')
    groups=[dict(params=actor.core.parameters(),lr=cfg['lr']),dict(params=actor.adapters.parameters(),lr=cfg['adapter_lr'])]
    opt=torch.optim.AdamW(groups,lr=cfg['lr'],weight_decay=0)
    dropped_cursor=next(i for i in range(100) if random.Random(cfg['seed']+i*130363+60261012).random()<.25)
    kept_cursor=next(i for i in range(100) if random.Random(cfg['seed']+i*130363+60261012).random()>=.25)
    losses=[]
    for step in range(3):
        examples=[];conditions=[];flags=[]
        for j,(ctx,dyn,acts,cond) in enumerate(cached):
            ex=sft_example(ctx.to(device),dyn.to(device),acts.to(device),20)
            cursor=dropped_cursor if (step+j)%2 else kept_cursor
            new,flag=apply_action_dropout(ex,cursor,cfg['seed'],.25)
            assert torch.equal(new['labels'],ex['labels'])
            examples.append(new);conditions.append({k:v.to(device) for k,v in cond.items()});flags.append(flag)
        assert sorted(flags)==[False,True]
        batch=_pad_examples(examples,actor.core.config.pad_token_id or 0)
        condition=_collate_condition(conditions)
        actor.train();opt.zero_grad(set_to_none=True)
        # Same warmup/cosine as the full trainer's first three steps.
        scale=(step+1)/cfg['warmup_steps']
        for g,lr in zip(opt.param_groups,(cfg['lr'],cfg['adapter_lr'])):g['lr']=lr*scale
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=actor(**batch,**condition,use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(actor.parameters(),1,error_if_nonfinite=True)
        null_grad=actor.core.get_input_embeddings().weight.grad[9006]
        assert torch.isfinite(null_grad).all() and null_grad.norm()>0
        opt.step()
        losses.append(dict(step=step+1,loss=float(loss.detach()),grad_norm=float(grad),dropped=flags))
        print(json.dumps(losses[-1]),flush=True)
    assert implementation()==before
    result=dict(state='PASS',device=device,optimizer_steps=3,micro_batch=2,checks=checks,
                zero_dropout_exact=True,labels_unchanged=True,nonaction_inputs_unchanged=True,
                null_embedding_received_gradient=True,losses=losses,
                feature_fingerprint=fh,extension_sources=source_identity(),
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                note='Three real single-GPU updates on cached train examples; not an 8-GPU DDP test or quality evaluation')
    path=EXP/'smoke_gpu.json';tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--memory-gib',type=float,default=6)
    a=p.parse_args();run(a.device,a.memory_gib)
