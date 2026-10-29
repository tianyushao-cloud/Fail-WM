"""Frozen train-only codec locality, regional NLL, and matched-action history probes."""
import argparse, hashlib, json, math, os, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT=Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT=ROOT/'diagnostics_step1_20261012'
CONFIG=ROOT/'experiments_mixed_codec_wm_20261011/natural.json'
RAW=Path('/media/tyshao/fail_WM/RoboCasa/data/batch1000/manifest.jsonl')
MODEL=ROOT/'evals_2x2_20261012/bundles/A'

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def frozen(p,x):
    if Path(p).exists():assert load(p)==x,p
    else:write(p,x)
def box(frame):
    x=frame.astype(np.float32);m=(x[...,0]>75)&(x[...,0]>1.5*x[...,1])&(x[...,0]>1.4*x[...,2])
    labels,n=ndimage.label(m);blobs=[]
    for j in range(1,n+1):
        y,xs=np.where(labels==j)
        if len(xs)>=8:blobs.append((len(xs),[int(xs.min()),int(y.min()),int(xs.max()+1),int(y.max()+1)]))
    if not blobs:return None
    blobs.sort(reverse=True);b=blobs[0][1]
    return dict(raw=b,expanded=[max(0,b[0]-8),max(0,b[1]-8),min(320,b[2]+8),min(256,b[3]+8)],components=len(blobs),area=blobs[0][0])
def region(b):
    m=np.zeros((256,320),bool)
    if b:
        x0,y0,x1,y1=b['expanded'];m[y0:y1,x0:x1]=True
    return m
def nominal_tokens(mask):return mask.reshape(8,32,10,32).any((1,3)).reshape(80)

def select():
    cfg=load(CONFIG);rows=load(cfg['train_manifest'])['cases'];raw={x['episode_id']:x for x in map(json.loads,RAW.read_text().splitlines())}
    used=set();pairs=[];frames=[]
    for wanted in ['skip_descend','grasp_offset','partial_close','early_release','early_release']:
        found=False
        for fail in sorted(rows,key=lambda x:x['id']):
            s=raw[fail['id']]
            if fail['success'] or fail['initial_state_id'] in used:continue
            if wanted not in [f['type'] for f in s['perturbation']['factors']]:continue
            if wanted!='early_release' and s['event_summary']['lift']:continue
            if wanted=='early_release' and not s['event_summary']['lift']:continue
            successes=[r for r in rows if r['success'] and r['initial_state_id']==fail['initial_state_id'] and not raw[r['id']]['perturbation']['factors']]
            if not successes:continue
            succ=sorted(successes,key=lambda x:x['id'])[0]
            Lf=s['phase_ranges']['attempt0/lift']['first_action']//4+1
            Ls=raw[succ['id']]['phase_ranges']['attempt0/lift']['first_action']//4+1
            anchor=s['phase_ranges']['attempt0/early_release']['first_action']//4+1 if wanted=='early_release' else Lf
            af=anchor;ass=Ls+(anchor-Lf)
            imgs={}
            for r in [fail,succ]:
                with np.load(Path(cfg['train_manifest']).parent/r['file']) as z:imgs[r['id']]=z['images'].copy()
            chosen=[]
            for offset in [2,5]:
                ft,st=af+offset,ass+offset
                if not all(8<=t<len(imgs[r['id']])-1 and box(imgs[r['id']][t]) for r,t in [(fail,ft),(succ,st)]):break
                chosen.append((ft,st))
            if len(chosen)!=2:continue
            pair=dict(index=len(pairs),factor=wanted,group=fail['initial_state_id'],failure=fail['id'],success=succ['id'],targets=chosen)
            pairs.append(pair);used.add(fail['initial_state_id'])
            for ft,st in chosen:
                for r,t,role in [(fail,ft,'failure'),(succ,st,'success')]:
                    frames.append(dict(key=f"{r['id']}_t{t:03d}",id=r['id'],target=t,role=role,pair=pair['index'],factor=wanted,box=box(imgs[r['id']][t])))
            found=True;break
        if not found:raise RuntimeError('No eligible pair for '+wanted)
    protocol=dict(state='FROZEN_BEFORE_MODEL_PROBES',pairs=pairs,frames=frames,seed=20261012,
        train_manifest_sha256=sha(cfg['train_manifest']),source_sha256=sha(RAW),bundle_manifest_sha256=sha(MODEL/'manifest.json'),
        script_sha256=sha(__file__),selection='Sorted first eligible cases per factor; five distinct train groups; GT ROI visible at phase+2/+5; same-group unperturbed success controls; no model-output selection',
        probes=['same-position other-selected-frame donor 1','same-position other-selected-frame donor 2','nearest different FSQ embedding'],
        donor_fallback='If selected 10 episodes have fewer than 2 distinct alternatives at a position, take ascending distinct codes observed at the same position in all800 training cache episodes; record fallback count',
        code_pool_sha256=sha(OUT/'train_code_pool.json'),
        precision='FP32 TF32off; original pair-aligned codec decoding; perturb one code at a time; 3 alternatives per token',
        pixel_influence='mean absolute RGB change against unmodified decode; minimal highest-energy pixel area covering 50/90 percent plus nominal tile/3x3 tile energy',
        history='fixed context/anchor/actions/text/target token teacher forcing; swap only last6 visual history frames with same-group opposite outcome phase-aligned histories; entire visual history may change arm and cube, diagnostic not physical counterfactual',
        limitations=['20 train frames/5 groups, exploratory mechanistic probe not generalization evaluation','Full-vocabulary teacher-forced CE is actual SFT token loss; RGB beam4 is free autoregression within the next frame only','ROI from largest red component expanded8; missing/fragmented detections disclosed','No WM optimization, no new simulator collection, no test access'])
    frozen(OUT/'protocol.json',protocol);print(json.dumps({'pairs':pairs,'frames':len(frames)}),flush=True)

def energy_stats(a,i,roi):
    total=float(a.sum());flat=np.sort(a.reshape(-1))[::-1];cum=flat.cumsum()
    y,x=divmod(i,10);tile=np.zeros_like(a,bool);tile[y*32:(y+1)*32,x*32:(x+1)*32]=True
    nearby=np.zeros_like(a,bool);nearby[max(0,y-1)*32:min(8,y+2)*32,max(0,x-1)*32:min(10,x+2)*32]=True
    py,px=np.unravel_index(int(a.argmax()),a.shape)
    return dict(mean_abs_rgb_change=total/a.size,area50_fraction=(int(np.searchsorted(cum,.5*total))+1)/a.size,
                area90_fraction=(int(np.searchsorted(cum,.9*total))+1)/a.size,
                nominal_tile_energy=float(a[tile].sum()/total),neighbor3x3_energy=float(a[nearby].sum()/total),
                roi_energy=float(a[roi].sum()/total),roi_absolute_energy=float(a[roi].sum()),peak_xy=[int(px),int(py)],
                peak_in_nominal_tile=bool(tile[py,px]),total=total) if total>1e-12 else dict(total=total,zero_effect=True)

def panel(path,frames,labels,roi=None):
    canvas=Image.new('RGB',(320*len(frames),280),'white');draw=ImageDraw.Draw(canvas)
    for i,(f,name) in enumerate(zip(frames,labels)):
        im=Image.fromarray(np.uint8(np.clip(f,0,255)));canvas.paste(im,(320*i,24));draw.text((320*i+4,4),name,fill='black')
        if roi:
            x0,y0,x1,y1=roi['expanded'];draw.rectangle((320*i+x0,24+y0,320*i+x1,24+y1),outline='yellow',width=1)
    canvas.save(path)

def run():
    import torch
    from futureworlds.pipeline import WorldPipeline
    from futureworlds.cache import load_cached
    from futureworlds.protocol import feature_fingerprint
    from futureworlds.memory import make_prefix,retained
    from futureworlds.decoding import beam_decode
    from futureworlds.batch import TensorBatch
    p=load(OUT/'protocol.json');assert p['script_sha256']==sha(__file__)
    cfg=load(CONFIG);assert sha(cfg['train_manifest'])==p['train_manifest_sha256']
    assert sha(OUT/'train_code_pool.json')==p['code_pool_sha256']
    code_pool=load(OUT/'train_code_pool.json')
    assert sha(MODEL/'manifest.json')==p['bundle_manifest_sha256']
    torch.set_num_threads(2);torch.manual_seed(p['seed']);np.random.seed(p['seed'])
    torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory)
    pipe=WorldPipeline(MODEL,'robocasa','sft','cuda',cfg['text_model']);pipe.model.eval()
    feature=feature_fingerprint(cfg['bundle'],'robocasa',cfg['text_model'])
    rows={r['id']:r for r in load(cfg['train_manifest'])['cases']};cache={};images={}
    for cid in sorted({f['id'] for f in p['frames']}):
        cache[cid]=load_cached(cfg['token_cache'],rows[cid],feature,'cuda')
        with np.load(Path(cfg['train_manifest']).parent/rows[cid]['file']) as z:images[cid]=z['images'].copy()
    # Forward with explicitly fixed cached text. Generation uses the same text via set_text.
    def set_text(cid):
        cond=cache[cid][3]
        pipe.model.set_text(cond['text_features'],cond['text_mask'])
    def prefix(cid,t,donor=None,dt=None):
        ctx,dyn,actions,_=cache[cid]
        hist={i:dyn[i:i+1] for i in retained(t)}
        if donor:
            dd=cache[donor][1]
            for i in retained(t):
                if i!=1:hist[i]=dd[dt+(i-t):dt+(i-t)+1]
        return make_prefix(ctx.reshape(1,-1)+4375,hist,actions,t)[0]
    @torch.inference_mode()
    def scores(pref,target):
        target=target.long()
        logits=pipe.model(input_ids=torch.cat([pref,target[:,:-1]],1),use_cache=False).logits[:,pref.shape[1]-1:].float()
        nll=-logits.log_softmax(-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)[0]
        visual=logits[0,:,:4375].log_softmax(-1)
        return nll.cpu().numpy(),visual
    @torch.inference_mode()
    def decode(cid,t,tokens):
        ctx,dyn,_,_=cache[cid];s=t-t%2;codes=tokens.reshape(1,1,80)
        if s<t:codes=torch.cat([dyn[s:t][None],codes],1)
        pixels=pipe.worker.detokenize(TensorBatch({'tokens':codes,'ctx_tokens':ctx}),None).batch['pixels'][0,1+t-s]
        return pixels.permute(1,2,0).cpu().numpy()*255
    def errors(pred,truth,roi):
        a=np.abs(pred.astype(np.float32)-truth.astype(np.float32)).mean(-1)
        return dict(roi_mae255=float(a[roi].mean()),outside_mae255=float(a[~roi].mean()),full_mae255=float(a.mean()))
    with torch.inference_mode():
        embeddings=pipe.codec.dynamics_quantize.indices_to_codes(torch.arange(4375,device='cuda')[None])[0]
        nearest={}
        for code in sorted({int(v) for f in p['frames'] for v in cache[f['id']][1][f['target']].tolist()}):
            dist=(embeddings-embeddings[code]).square().sum(-1);dist[code]=torch.inf;nearest[code]=int(dist.argmin())
    for index,f in enumerate(p['frames']):
        dest=OUT/'frames'/f['key'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():continue
        started=time.time();cid,t=f['id'],f['target'];ctx,dyn,acts,cond=cache[cid]
        gt=images[cid][t];roi=region(f['box']);nominal=nominal_tokens(roi)
        # Three code changes per spatial position. Same-position donors from fixed frame list.
        alternatives=[];fallback_positions=[]
        for i,code in enumerate(dyn[t].tolist()):
            donors=[]
            for other in p['frames'][index+1:]+p['frames'][:index+1]:
                v=int(cache[other['id']][1][other['target'],i])
                if v!=code and v not in donors:donors.append(v)
                if len(donors)==2:break
            # Rare invariant positions: use other real train times in selected episodes.
            if len(donors)<2:
                for data in cache.values():
                    for v in data[1][:,i].tolist():
                        if v!=code and v not in donors:donors.append(v)
                        if len(donors)==2:break
                    if len(donors)==2:break
            if len(donors)<2:
                fallback_positions.append(i)
                for v in code_pool[i]:
                    if v!=code and v not in donors:donors.append(v)
                    if len(donors)==2:break
            if len(donors)<2:raise ValueError('Need two distinct observed donors')
            alternatives.append(donors+[nearest[int(code)]])
        s=t-t%2;pair=dyn[s:s+2][None].clone();offset=t-s
        with torch.inference_mode():
            raw=pipe.codec.detokenize(ctx,pair).float().clamp(0,1)
            baseline=raw[0,1+offset].permute(1,2,0).cpu().numpy()
            # Repeated batch must agree with deployed serial worker before batching probes.
            worker=pipe.worker.detokenize(TensorBatch({'tokens':pair,'ctx_tokens':ctx}),None).batch['pixels']
            assert torch.allclose(raw,worker,atol=1e-6,rtol=0)
            bat=pipe.codec.detokenize(ctx.expand(4,-1,-1),pair.expand(4,-1,-1)).float().clamp(0,1)
            batch_error=float((bat-raw).abs().max());assert batch_error<2e-4,batch_error
        influence=np.zeros((80,256,320),np.float32);records=[];per_kind=np.zeros((3,80),np.float64)
        neighbor_max=0.0
        jobs=[(i,j,v) for i,vs in enumerate(alternatives) for j,v in enumerate(vs)]
        for start in range(0,len(jobs),4):
            chunk=jobs[start:start+4];modified=pair.expand(len(chunk),-1,-1).clone()
            for n,(i,j,v) in enumerate(chunk):modified[n,offset,i]=v
            with torch.inference_mode():
                pix=pipe.codec.detokenize(ctx.expand(len(chunk),-1,-1),modified).float().clamp(0,1)
            neighbor_max=max(neighbor_max,float((pix[:,1+(1-offset)]-raw[:,1+(1-offset)]).abs().max()))
            target=pix[:,1+offset].permute(0,2,3,1).cpu().numpy()
            for a,(i,j,v) in zip(target,chunk):
                energy=np.abs(a-baseline).mean(-1);influence[i]+=energy/3
                st=energy_stats(energy,i,roi);records.append(dict(token=i,kind=j,original=int(dyn[t,i]),replacement=v,**st));per_kind[j,i]=st.get('roi_absolute_energy',0)
        roi_imp=influence[:,roi].sum(-1);order=np.argsort(roi_imp)[::-1];cs=roi_imp[order].cumsum();total=float(cs[-1])
        n50=int(np.searchsorted(cs,total*.5))+1;n90=int(np.searchsorted(cs,total*.9))+1
        empirical=np.zeros(80,bool);empirical[order[:n90]]=True
        averaged=[dict(token=i,**energy_stats(a,i,roi)) for i,a in enumerate(influence)]
        set_text(cid);pref=prefix(cid,t)
        nll,visual=scores(pref,dyn[t:t+1]);del visual
        with torch.inference_mode():beam,_,_=beam_decode(pipe.model,pref,width=4)
        pred=decode(cid,t,beam[:1]);recon=baseline*255
        def token_summary(mask):
            return dict(tokens=int(mask.sum()),mean_nll=float(nll[mask].mean()),outside_mean_nll=float(nll[~mask].mean()) if (~mask).any() else None,
                        loss_share=float(nll[mask].sum()/nll.sum()))
        result=dict(frame=f,nominal_roi=token_summary(nominal),empirical_roi90=token_summary(empirical),
                    roi_area_fraction=float(roi.mean()),roi_influence_n50=n50,roi_influence_n90=n90,
                    nominal_roi_influence_share=float(roi_imp[nominal].sum()/total),top_roi_tokens=order.tolist(),
                    replacement_kind_roi_rank_correlation=np.corrcoef(per_kind).tolist(),
                    locality=averaged,probe_records=records,all_train_donor_fallback_positions=fallback_positions,batch_vs_serial_max_pixel_error=batch_error,
                    other_frame_max_pixel_change=neighbor_max,all_token_nll=float(nll.mean()),
                    codec_rgb=errors(recon,gt,roi),wm_beam_nextframe_rgb=errors(pred,gt,roi),
                    wm_vs_codec_rgb=errors(pred,recon,roi),seconds=time.time()-started)
        np.savez_compressed(dest/'arrays.npz',influence=influence.astype(np.float16),nll=nll,roi_influence=roi_imp,beam_tokens=beam[:1].cpu().numpy(),per_kind_roi=per_kind)
        panel(dest/'gt_codec_nextframe.jpg',[gt,recon,pred],['GT','GT codec','A next frame beam4'],f['box'])
        Image.fromarray(np.uint8(np.clip(pred,0,255))).save(dest/'beam.png')
        write(dest/'result.json',result);print(json.dumps({'phase':'locality_nll','frame':f['key'],'done':index+1,'total':20,'seconds':result['seconds']}),flush=True)
    for pair_info in p['pairs']:
        for ft,st in pair_info['targets']:
            key=f"pair{pair_info['index']}_f{ft}_s{st}";dest=OUT/'history'/key;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():continue
            fail,succ=pair_info['failure'],pair_info['success'];result=dict(pair=pair_info,targets=[ft,st],directions=[])
            for base,t,donor,dt in [(fail,ft,succ,st),(succ,st,fail,ft)]:
                set_text(base);own=prefix(base,t);swapped=prefix(base,t,donor,dt)
                # Every action segment and fixed context/anchor are bit-identical.
                assert torch.equal(own[:,:1280+80],swapped[:,:1280+80])
                for j in range(len(retained(t))):assert torch.equal(own[:,1280+j*132+80:1280+(j+1)*132],swapped[:,1280+j*132+80:1280+(j+1)*132])
                local=[]
                for target_owner,target_t in [(base,t),(donor,dt)]:
                    target=cache[target_owner][1][target_t:target_t+1]
                    n1,lp1=scores(own,target);n2,lp2=scores(swapped,target)
                    mid=torch.logaddexp(lp1,lp2)-math.log(2)
                    js=.5*((lp1.exp()*(lp1-mid)).sum(-1)+(lp2.exp()*(lp2-mid)).sum(-1))
                    union=nominal_tokens(region(box(images[base][t])))|nominal_tokens(region(box(images[donor][dt])))
                    local.append(dict(target_owner=target_owner,mean_nll_own_history=float(n1.mean()),mean_nll_swapped_history=float(n2.mean()),
                        nll_swapped_minus_own=float((n2-n1).mean()),roi_union_nll_swapped_minus_own=float((n2-n1)[union].mean()),
                        visual_js_mean=float(js.mean()),visual_js_first_token=float(js[0]),visual_js_roi_union=float(js.cpu().numpy()[union].mean()),
                        teacherforced_argmax_changed_fraction=float((lp1.argmax(-1)!=lp2.argmax(-1)).float().mean())))
                    del lp1,lp2,mid,js
                # Determinism control on a repeated own-history forward, no dropout.
                n1,_=scores(own,cache[base][1][t:t+1]);nr,_=scores(own,cache[base][1][t:t+1]);assert np.array_equal(n1,nr)
                with torch.inference_mode():changed,_,_=beam_decode(pipe.model,swapped,width=4)
                own_frame=load(OUT/'frames'/f'{base}_t{t:03d}'/'result.json')
                with np.load(OUT/'frames'/f'{base}_t{t:03d}'/'arrays.npz') as z:own_codes=z['beam_tokens'].copy()
                own_pixels=np.asarray(Image.open(OUT/'frames'/f'{base}_t{t:03d}'/'beam.png'))
                swapped_pixels=decode(base,t,changed[:1]);roi=region(box(images[base][t]))|region(box(images[donor][dt]))
                panel(dest/f'{base}.jpg',[images[base][t-1],images[donor][dt-1],images[base][t],own_pixels,swapped_pixels],
                      ['Own last history','Donor last history','Own GT target','Own history -> next','Swapped history -> next'])
                result['directions'].append(dict(base=base,donor=donor,target=t,donor_target=dt,
                    history_visual_token_changed_fraction=float((own!=swapped).sum()/ (6*80)),
                    actions_context_anchor_unchanged=True,repeat_nll_exact=True,conditional_scores=local,
                    free_beam_token_changed_fraction=float((own_codes!=changed[:1].cpu().numpy()).mean()),
                    free_beam_image_change=errors(swapped_pixels,own_pixels,roi),
                    own_gt_box=box(images[base][t]),donor_gt_box=box(images[donor][dt]),
                    own_beam_box=box(own_pixels),swapped_beam_box=box(np.uint8(np.clip(swapped_pixels,0,255)))))
            write(dest/'result.json',result);print(json.dumps({'phase':'history','pair':key}),flush=True)
    write(OUT/'complete.json',dict(state='COMPLETE',frames=20,token_interventions=4800,history_pairs=10,history_directions=20,model=str(MODEL),peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','run']);a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    {'select':select,'run':run}[a.phase]()
