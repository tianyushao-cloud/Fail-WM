#!/usr/bin/env python3
"""Read-only model diagnostics: validation NLL, matched actions, codec ROI, sampling."""
import argparse
import gc
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from futureworlds.batch import TensorBatch
from futureworlds.checkpoints import file_sha256
from futureworlds.data import read_case, read_manifest
from futureworlds.memory import make_prefix, retained, rollout
from futureworlds.pipeline import WorldPipeline
from robocasa_video_judge import calibrate, video_features

ROOT = Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT = ROOT / 'evals_diagnostics_20261009'
MANIFEST = ROOT / 'robocasa_data/val.json'
TEXT = '/media/tyshao/fail_WM/FutureWorlds_workspace/assets/t5-base'
EVENTS = {'contact', 'grasp', 'lift', 'drop', 'release', 'grasp_attempt_failed'}


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def mean(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def red(frame):
    x = frame.astype(float)
    return (x[..., 0] > 75) & (x[..., 0] > 1.5*x[..., 1]) & (x[..., 0] > 1.4*x[..., 2])


def center(frame):
    y, x = np.where(red(frame))
    return [float(x.mean()), float(y.mean())] if len(x) >= 8 else None


def roi(frame):
    mask = np.zeros((256, 320), dtype=bool)
    y, x = np.where(red(frame))
    if len(x) >= 8:
        mask[max(0,y.min()-8):min(256,y.max()+9),max(0,x.min()-8):min(320,x.max()+9)] = True
    return mask


def roi_tokens(frame):
    # Spatially aligned 8x10 grid; this is a coarse proxy, not an exact receptive field.
    return roi(frame).reshape(8,32,10,32).any(axis=(1,3)).reshape(80)


def targets(row, raw):
    events = [{'name': e['event'], 'raw_state_index': int(e['state_index']),
               'target': math.ceil(e['state_index']/4)} for e in raw['events']
              if e['event'] in EVENTS and 'state_index' in e]
    ts = sorted({i for e in events for i in range(e['target']-2,e['target']+5)
                 if 2 <= i < row['frames']})
    return events, ts


def load_features(cid):
    return torch.load(OUT / 'features' / (cid+'.pt'), weights_only=True, map_location='cpu')


def encode_video(frames, path, labels=None):
    rgb = np.asarray(frames, dtype=np.uint8)
    h,w = rgb.shape[1:3]
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
                    '-pix_fmt','rgb24','-s',f'{w}x{h}','-r','5','-i','pipe:0',
                    '-an','-c:v','libx264','-crf','14','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(path)], input=rgb.tobytes(),check=True,capture_output=True)


def prepare():
    rows = read_manifest(MANIFEST,'robocasa')['cases']
    raw = {r['episode_id']:r for r in map(json.loads, Path('/media/tyshao/fail_WM/RoboCasa/data/batch1000/manifest.jsonl').read_text().splitlines())}
    (OUT/'features').mkdir(exist_ok=True)
    pipe = WorldPipeline(OUT/'natural_bundle','robocasa','sft','cuda',TEXT)
    records, real_judge, selected = [], [], []
    for index,row in enumerate(rows):
        images, actions, text = read_case(MANIFEST,row,52)
        events, ts = targets(row,raw[row['id']])
        # Entire sequence is tokenized with the same first-frame context as SFT.
        with torch.inference_mode():
            ctx,dyn,act,_,cond = pipe.encode_training_window(images,actions,text)
        torch.save({'context':ctx.cpu(), 'dynamics':dyn.cpu(), 'actions':act.cpu(),
                    'condition':{k:v.cpu() for k,v in cond.items()},
                    'events':events,'targets':ts,'source_sha256':row['sha256']},
                   OUT/'features'/(row['id']+'.pt'))
        positions=sorted(set(ts+[len(images)-1]))
        for begin in range(0,len(positions),8):
            inds=positions[begin:begin+8]
            with torch.inference_mode():
                decoded=pipe.codec.detokenize(ctx,dyn[inds][None]).float().clamp(0,1)[0,1:]
            pred=np.rint(decoded.permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
            for t,recon in zip(inds,pred):
                truth=images[t]; mask=roi(truth); delta=np.abs(recon.astype(float)-truth.astype(float))
                a,b=center(truth),center(recon)
                rec={'id':row['id'],'target':t,'success':row['success'],
                     'failure_type':row['failure_type'],'event_window':t in ts,
                     'full_mae_255':float(delta.mean()),
                     'cube_roi_mae_255':float(delta[mask].mean()) if mask.any() else None,
                     'real_cube_visible':a is not None,'reconstructed_cube_visible':b is not None,
                     'cube_center_error_px':float(np.linalg.norm(np.array(a)-np.array(b))) if a and b else None}
                records.append(rec)
                if mask.any() and (rec['cube_roi_mae_255']>15 or t==len(images)-1):
                    panel=Image.new('RGB',(640,280),'black'); panel.paste(Image.fromarray(truth),(0,24));panel.paste(Image.fromarray(recon),(320,24))
                    ImageDraw.Draw(panel).text((6,5),f"{row['id']} t={t} REAL | CODEC ROI MAE={rec['cube_roi_mae_255']:.2f}",fill='white')
                    selected.append((rec['cube_roi_mae_255'],panel))
        real_judge.append(dict(id=row['id'],success=row['success'],**video_features(images[2:])))
        if (index+1)%10==0: print('codec/features',index+1,len(rows),flush=True)
    groups={}
    for kind in ['success','failure']+sorted({r['failure_type'] for r in rows if not r['success']}):
        rr=[r for r in records if (r['success'] if kind=='success' else not r['success'] if kind=='failure' else r['failure_type']==kind)]
        by=defaultdict(list)
        for r in rr:by[r['id']].append(r)
        groups[kind]={'episodes':len(by),'frames':len(rr),
                      'episode_mean_roi_mae_255':mean([mean([r['cube_roi_mae_255'] for r in a]) for a in by.values()]),
                      'episode_mean_center_error_px':mean([mean([r['cube_center_error_px'] for r in a]) for a in by.values()]),
                      'missed_visible_cube_frames':sum(r['real_cube_visible'] and not r['reconstructed_cube_visible'] for r in rr)}
    for i,(score,panel) in enumerate(sorted(selected,key=lambda x:x[0],reverse=True)[:12]):panel.save(OUT/f'codec_worst_{i:02d}.png')
    write(OUT/'codec_audit.json',{'split':'val','groups':groups,'frames':records,
                               'roi_definition':'GT red-cube bounding box padded 8 pixels; absent cube -> unavailable, not zero error'})
    write(OUT/'real_judge_calibration.json',{'calibration':calibrate(real_judge),'rows':real_judge,
                                          'limitation':'RGB endpoint proxy; generated-video accuracy is unverified.'})
    write(OUT/'provenance.json',{'manifest':str(MANIFEST),'manifest_sha256':file_sha256(MANIFEST),
                               'models':{m:file_sha256(OUT/f'{m}_bundle'/'manifest.json') for m in ('success','natural')},
                               'event_mapping':'ceil(raw state_index/4); targets -2..+4 inclusive',
                               'samples':len(rows),'implementation_sha256':file_sha256(Path(__file__))})


@torch.inference_mode()
def score(model, prefix, response, mask):
    response = response.long()
    ids=torch.cat([prefix,response[:,:-1]],1)
    p=prefix.shape[1]
    logits=model(input_ids=ids,use_cache=False).logits[:,p-1:p+79].float()
    lp=logits[:,:,:4375].log_softmax(-1).gather(-1,response.unsqueeze(-1))[0,:,0]
    full=logits.log_softmax(-1).gather(-1,response.unsqueeze(-1))[0,:,0]
    v=lp.cpu().numpy(); f=full.cpu().numpy()
    return {'visual_nll_per_token':float(-v.mean()),'training_vocab_nll_per_token':float(-f.mean()),
            'cube_grid_nll_per_token':float(-v[mask].mean()) if mask.any() else None}


def pairs():
    rows=read_manifest(MANIFEST,'robocasa')['cases']
    by_state=defaultdict(list)
    for r in rows:
        if r['success']:by_state[r['initial_state_sha256']].append(r)
    selected=[]
    (OUT/'pair_features').mkdir(exist_ok=True)
    pipe=WorldPipeline(OUT/'natural_bundle','robocasa','sft','cuda',TEXT)
    for r in rows:
        if r['success'] or not by_state[r['initial_state_sha256']]:continue
        s=sorted(by_state[r['initial_state_sha256']],key=lambda x:x['id'])[0]
        fa,fb=load_features(s['id']),load_features(r['id'])
        ia,_,_=read_case(MANIFEST,s,52); ib,_,_=read_case(MANIFEST,r,52)
        initial_mae=float(np.abs(ia[:2].astype(float)-ib[:2].astype(float)).mean())
        if initial_mae>0.01:
            selected.append({'success_id':s['id'],'failure_id':r['id'],'eligible':[], 'excluded':'initial_images_differ'});continue
        # Both futures use exactly the same initial context image. Their genuine
        # future frames remain unchanged, including the failure's anchor frame.
        common=ib.copy();common[0]=ia[0]
        _,raw_actions,instruction=read_case(MANIFEST,r,52)
        with torch.inference_mode():
            ctx,dyn,act,_,cond=pipe.encode_training_window(common,raw_actions,instruction)
        fb=dict(context=ctx.cpu(),dynamics=dyn.cpu(),actions=act.cpu(),
                condition={k:v.cpu() for k,v in cond.items()})
        if not torch.equal(fa['context'],fb['context']):raise ValueError('Shared codec context mismatch')
        torch.save(fb,OUT/'pair_features'/(r['id']+'.pt'))
        eligible=[]; rejected=0
        for t in range(2,min(len(ia),len(ib))):
            kept=retained(t)
            if torch.equal(fa['actions'][kept],fb['actions'][kept]):continue
            history_mae=max(float(np.abs(ia[k].astype(float)-ib[k].astype(float)).mean()) for k in kept)
            agreement=float((fa['dynamics'][kept]==fb['dynamics'][kept]).float().mean())
            if history_mae>0.5 or agreement<0.95:
                rejected+=1; continue
            eligible.append({'target':t,'history_mae_255_max':history_mae,'history_token_agreement':agreement,
                             'different_target_tokens':int((fa['dynamics'][t]!=fb['dynamics'][t]).sum())})
        # Predeclare bounded evaluation: earliest and latest three eligible targets.
        chosen=[x for i,x in enumerate(eligible) if i<3 or i>=len(eligible)-3]
        selected.append({'success_id':s['id'],'failure_id':r['id'],'failure_type':r['failure_type'],
                      'shared_codec_context':True,'initial_two_frame_mae_255':initial_mae,
                      'eligible':chosen,'eligible_count':len(eligible),'rejected_history_count':rejected})
    write(OUT/'e0_pair_selection.json',selected)
    print('Shared-context pairs',len(selected),'eligible',sum(bool(x['eligible']) for x in selected),flush=True)


def e0():
    rows=read_manifest(MANIFEST,'robocasa')['cases']
    index={r['id']:r for r in rows}
    selected=json.loads((OUT/'e0_pair_selection.json').read_text())
    if any(p['eligible'] and not p.get('shared_codec_context') for p in selected):
        raise ValueError('Run pairs phase to normalize initial codec context first')
    for name in ('success','natural'):
        pipe=WorldPipeline(OUT/f'{name}_bundle','robocasa','sft','cuda',TEXT)
        pipe.model.eval(); swaps=[]
        previous=OUT/f'e0_{name}.json'
        records=json.loads(previous.read_text())['nll'] if previous.exists() else []
        for i,row in enumerate(rows if not records else []):
            f=load_features(row['id']);images,_,_=read_case(MANIFEST,row,52)
            ctx=f['context'].cuda(); dyn=f['dynamics'].cuda();act=f['actions'].cuda()
            pipe.model.set_text(f['condition']['text_features'].cuda(),f['condition']['text_mask'].cuda())
            ts=f['targets']
            # Four deterministic non-event targets provide a same-episode comparison.
            other=[t for t in range(2,len(dyn)) if t not in ts]
            controls=sorted({other[j] for j in np.linspace(0,len(other)-1,min(4,len(other))).astype(int)}) if other else []
            for t in sorted(set(ts+controls)):
                prefix,_=make_prefix(ctx.reshape(1,-1)+4375,{k:dyn[k:k+1] for k in retained(t)},act,t)
                records.append(dict(id=row['id'],success=row['success'],failure_type=row['failure_type'],target=t,
                                    event_window=t in ts,**score(pipe.model,prefix,dyn[t:t+1],roi_tokens(images[t]))))
            if (i+1)%20==0:print('E0 NLL',name,i+1,len(rows),flush=True)
        for pair in selected:
            if not pair['eligible']:continue
            a=load_features(pair['success_id'])
            b=torch.load(OUT/'pair_features'/(pair['failure_id']+'.pt'),weights_only=True,map_location='cpu')
            ia,_,_=read_case(MANIFEST,index[pair['success_id']],52)
            ib,_,_=read_case(MANIFEST,index[pair['failure_id']],52)
            ctx=a['context'].cuda(); dyn=a['dynamics'].cuda()
            pipe.model.set_text(a['condition']['text_features'].cuda(),a['condition']['text_mask'].cuda())
            for e in pair['eligible']:
                t=e['target']; scores={}
                for action_name,f in [('success',a),('failure',b)]:
                    pre,_=make_prefix(ctx.reshape(1,-1)+4375,{k:dyn[k:k+1] for k in retained(t)},f['actions'].cuda(),t)
                    for target_name,d,im in [('success',a,ia),('failure',b,ib)]:
                        scores[f'{target_name}_given_{action_name}']=score(pipe.model,pre,d['dynamics'][t:t+1].cuda(),roi_tokens(im[t]))['visual_nll_per_token']
                swaps.append(dict(success_id=pair['success_id'],failure_id=pair['failure_id'],failure_type=pair['failure_type'],**e,**scores,
                                  failure_action_preference=scores['failure_given_success']-scores['failure_given_failure'],
                                  success_action_preference=scores['success_given_failure']-scores['success_given_success'],
                                  paired_selectivity=scores['failure_given_success']+scores['success_given_failure']-scores['failure_given_failure']-scores['success_given_success']))
        write(OUT/f'e0_{name}.json',{'nll':records,'swaps':swaps,
                                  'swap_note':'Common success visual history; raw history MAE<=0.5 and visual token agreement>=0.95; positive preference favors matched action. Diagnostic tolerances, not proof of physical equivalence.'})
        print('E0 done',name,'targets',len(records),'swaps',len(swaps),flush=True)
        del pipe;gc.collect();torch.cuda.empty_cache()


def e1():
    rows=read_manifest(MANIFEST,'robocasa')['cases']
    # Fixed before looking at E0 scores: first validation failure in each of two event types.
    chosen=[next(r for r in rows if r['failure_type']==kind) for kind in ('grasp_miss','dropped_before_goal')]
    threshold=json.loads((OUT/'real_judge_calibration.json').read_text())['calibration']['threshold_pixels']
    report={'selection':'First validation grasp_miss and dropped_before_goal, independent of E0 scores',
            'temperature':1.0,'samples_per_case':8,'seeds':[2026100900,2026100901],
            'threshold_pixels':threshold,'judge':'provisional RGB endpoint proxy; missing cube is indeterminate','cases':[]}
    for name in ('success','natural'):
        pipe=WorldPipeline(OUT/f'{name}_bundle','robocasa','sft','cuda',TEXT)
        for row in chosen:
            images,actions,text=read_case(MANIFEST,row,52); horizon=len(images)-2
            dest=OUT/'e1'/name/row['id'];dest.mkdir(parents=True,exist_ok=True)
            result={'model':name,'id':row['id'],'failure_type':row['failure_type'],'horizon':horizon,'predictions':[]}
            encode_video(images[2:],dest/'real.mp4')
            with torch.inference_mode():
                beam=pipe.predict(images[:2],actions,text,horizon,4,'full')['prediction']
            movies=[('beam4',np.rint(beam*255).astype(np.uint8))]
            f=load_features(row['id']);ctx,anchor=pipe.encode_observations(images[:2]);acts=f['actions'].cuda();pipe.condition(text)
            for batch in range(2):
                generator=torch.Generator(device='cuda').manual_seed(2026100900+batch)
                with torch.inference_mode():
                    frames,_=rollout(pipe.model,ctx.reshape(1,-1)+4375,anchor,acts,horizon,4,generator,temperature=1.0)
                    codes=torch.stack([x['response'] for x in frames],1)
                    pix=pipe.worker.detokenize(TensorBatch({'tokens':codes,'ctx_tokens':ctx.expand(4,-1,-1)}),None).batch['pixels'][:,1:]
                rgb=np.rint(pix.permute(0,1,3,4,2).cpu().numpy()*255).astype(np.uint8)
                for j in range(4):movies.append((f'sample_{batch*4+j:02d}',rgb[j]))
                del frames,codes,pix
                print('E1 samples',name,row['id'],(batch+1)*4,flush=True)
            for label,movie in movies:
                features=video_features(movie)
                # Require detection in >=3 of final 5 frames and minimum blob sizes.
                usable=features['observed']>=3 and features['red_area_median']>=8 and features['green_area_median']>=20
                outcome=('success' if features['distance_median']<threshold else 'failure') if usable else 'indeterminate'
                result['predictions'].append({'name':label,'proxy_outcome':outcome,**features})
                encode_video(movie,dest/(label+'.mp4'))
            sheet=Image.new('RGB',(960,3*280),'black');draw=ImageDraw.Draw(sheet)
            for i,(label,movie) in enumerate(movies):
                x,y=(i%3)*320,(i//3)*280;sheet.paste(Image.fromarray(movie[-1]),(x,y+24))
                outcome=result['predictions'][i]['proxy_outcome'];draw.text((x+4,y+5),label+' '+outcome,fill='white')
            sheet.save(dest/'last_frames.png')
            write(dest/'result.json',result);report['cases'].append(result);write(OUT/'e1_summary.json',report)
            print('E1 done',name,row['id'],Counter(r['proxy_outcome'] for r in result['predictions'][1:]),flush=True)
        del pipe;gc.collect();torch.cuda.empty_cache()


def summarize():
    lines=['# World-model diagnostics','', 'Validation-only diagnostics; no training performed.','']
    for name in ('success','natural'):
        data=json.loads((OUT/f'e0_{name}.json').read_text()); rec=data['nll']; swaps=data['swaps']
        lines.append(f'## {name}')
        for success in (True,False):
            for event in (True,False):
                rr=[r for r in rec if r['success']==success and r['event_window']==event]
                by=defaultdict(list)
                for r in rr:by[r['id']].append(r)
                lines.append(f'- success={success}, event={event}, episodes={len(by)}, targets={len(rr)}, episode-mean visual NLL={mean([mean([r["visual_nll_per_token"] for r in z]) for z in by.values()])}')
        informative=[r for r in swaps if r['different_target_tokens']>0]
        by=defaultdict(list)
        for r in informative:by[r['failure_id']].append(r)
        means=[mean([r['failure_action_preference'] for r in z]) for z in by.values()]
        lines.append(f'- Action-swap informative targets={len(informative)}, pairs={len(by)}, mean paired failure preference={mean(means)}, positive pairs={sum(x>0 for x in means)}')
    lines += ['', 'Codec summaries:', json.dumps(json.loads((OUT/'codec_audit.json').read_text())['groups'],indent=2),
              '', 'NLL does not measure optimism. RGB sampling outcomes are provisional; absent/low-area cube is indeterminate.',
              'A lower full-image error or an occasional failed sample does not prove calibrated outcome prediction.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def verify():
    from futureworlds.training import sft_example
    rows=read_manifest(MANIFEST,'robocasa')['cases']
    pipe=WorldPipeline(OUT/'natural_bundle','robocasa','sft','cuda',TEXT)
    pipe.model.eval()
    row=next(r for r in rows if r['failure_type']=='grasp_miss')
    f=load_features(row['id']);t=f['targets'][0]
    ctx,dyn,act=(f[k].cuda() for k in ('context','dynamics','actions'))
    pipe.model.set_text(f['condition']['text_features'].cuda(),f['condition']['text_mask'].cuda())
    example=sft_example(ctx,dyn.long(),act,t)
    prefix,_=make_prefix(ctx.reshape(1,-1)+4375,{k:dyn[k:k+1] for k in retained(t)},act,t)
    with torch.inference_mode():
        reference=float(pipe.model(**example,use_cache=False).loss)
        calculated=score(pipe.model,prefix,dyn[t:t+1],np.ones(80,dtype=bool))['training_vocab_nll_per_token']
    if abs(reference-calculated)>1e-4:raise ValueError('NLL does not match training loss')
    threshold=json.loads((OUT/'real_judge_calibration.json').read_text())['calibration']['threshold_pixels']
    results=[]
    for row in rows:
        images,_,_=read_case(MANIFEST,row,52); f=load_features(row['id'])
        with torch.inference_mode():
            recon=pipe.codec.detokenize(f['context'].cuda(),f['dynamics'][-5:][None].cuda()).float().clamp(0,1)[0,1:]
        rgb=np.rint(recon.permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
        r={'id':row['id'],'true_success':row['success'],'failure_type':row['failure_type']}
        for name,frames in [('real',images[-5:]),('codec',rgb)]:
            v=video_features(frames)
            usable=v['observed']>=3 and v['red_area_median']>=8 and v['green_area_median']>=20
            v['outcome']=('success' if v['distance_median']<threshold else 'failure') if usable else 'indeterminate'
            r[name]=v
        results.append(r)
    write(OUT/'verification.json',{'training_loss':reference,'diagnostic_full_vocab_nll':calculated,
                                 'absolute_difference':abs(reference-calculated),
                                 'endpoint_codec_proxy_audit':results,
                                 'script_sha256':file_sha256(Path(__file__))})
    print('NLL verified',reference,calculated,flush=True)
    print('codec endpoint transitions',Counter((r['true_success'],r['real']['outcome'],r['codec']['outcome']) for r in results),flush=True)


def sampling_audit():
    rows={r['id']:r for r in read_manifest(MANIFEST,'robocasa')['cases']}
    summary=json.loads((OUT/'e1_summary.json').read_text())
    checks=[]
    for case in summary['cases']:
        images,_,_=read_case(MANIFEST,rows[case['id']],52)
        truth=images[2:]; tc=[center(x) for x in truth]
        dest=OUT/'e1'/case['model']/case['id']
        for item in case['predictions']:
            target=dest/(item['name']+'.mp4')
            raw=subprocess.run(['ffmpeg','-v','error','-i',str(target),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],check=True,capture_output=True).stdout
            decoded=np.frombuffer(raw,dtype=np.uint8).reshape(-1,256,320,3)
            if len(decoded)!=len(truth):raise ValueError('Decoded prediction length mismatch')
            pc=[center(x) for x in decoded]
            distances=[float(np.linalg.norm(np.array(a)-np.array(b))) if a is not None and b is not None else None for a,b in zip(tc,pc)]
            checks.append({'model':case['model'],'id':case['id'],'name':item['name'],
                           'proxy_outcome':item['proxy_outcome'],'verified_frames':len(decoded),
                           'mean_cube_error_px_when_visible':mean(distances),
                           'compared_frames':sum(x is not None for x in distances),
                           'true_visible_pred_missing_frames':sum(a is not None and b is None for a,b in zip(tc,pc)),
                           'note':'Cube tracking on decoded MP4s; RGB proxy, not simulator validation'})
    write(OUT/'sampling_tracking_audit.json',checks)
    print('Verified MP4s',len(checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','pairs','e0','e1','summarize','verify','sampling_audit'));args=parser.parse_args()
    OUT.mkdir(exist_ok=True);globals()[args.phase]()
