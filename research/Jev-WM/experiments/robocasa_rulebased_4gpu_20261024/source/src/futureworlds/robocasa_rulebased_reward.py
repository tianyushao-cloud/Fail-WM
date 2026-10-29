"""Pure rule-based reward: position + persistence only, no VLM judge."""
import time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from .batch import TensorBatch
from .rewards import CodecWorker
from .data import write_json


def track(clip):
    out=[]
    for frame in clip:
        x=frame.astype(np.float32)
        mask=(x[...,0]>75)&(x[...,0]>1.5*x[...,1])&(x[...,0]>1.4*x[...,2])
        labels,n=ndimage.label(mask)
        blobs=[]
        for k in range(1,n+1):
            yy,xx=np.where(labels==k)
            if len(xx)>=6: blobs.append(dict(xy=[float(xx.mean()),float(yy.mean())],area=len(xx)))
        blobs.sort(key=lambda b:b['area'],reverse=True)
        out.append(dict(xy=blobs[0]['xy'] if blobs else None,area=sum(b['area'] for b in blobs),count=len(blobs)))
    return out


def aggregate(errors):
    """Bad transient frames retain weight even in long mostly-correct clips."""
    a=np.asarray(errors,dtype=float)
    return float(.5*a.mean()+.5*np.sort(a)[-max(1,int(np.ceil(len(a)/4))):].mean())


def rule_scores(reference,prediction):
    gt,pr=track(reference),track(prediction)
    pos=[];exist=[];motion=[]
    for g,p in zip(gt,pr):
        if g['xy'] is not None:
            d=np.linalg.norm(np.asarray(g['xy'])-p['xy']) if p['xy'] is not None else float('inf')
            pos.append(float(1-np.exp(-max(0,d-2)/20)))
            if p['xy'] is None:exist.append(1.)
            else:
                ratio=min(g['area'],p['area'])/max(g['area'],p['area'])
                area=max(0.,(0.6-ratio)/0.6)
                duplicate=float(p['count']>g['count'])
                exist.append(max(area,duplicate))
        else:
            exist.append(.25 if p['xy'] is not None else 0.)
    for i in range(1,len(gt)):
        ga,gb,pa,pb=gt[i-1],gt[i],pr[i-1],pr[i]
        if ga['xy'] is not None and gb['xy'] is not None:
            if pa['xy'] is None or pb['xy'] is None:motion.append(1.)
            else:
                dg=np.asarray(gb['xy'])-ga['xy'];dp=np.asarray(pb['xy'])-pa['xy']
                motion.append(float(1-np.exp(-max(0,np.linalg.norm(dp-dg)-2)/12)))
    position=1-aggregate(pos) if pos else .5
    persistence=1-(.7*aggregate(exist)+.3*(aggregate(motion) if motion else 0.))
    return dict(position=position,persistence=persistence),dict(gt=gt,pred=pr,visible_GT_frames=len(pos),exist_error=aggregate(exist),motion_error=aggregate(motion) if motion else 0.)


class RoboCasaRuleBasedReward:
    """Pure rule-based reward without VLM judge."""
    def __init__(self,codec,config,device,horizon):
        self.worker=CodecWorker(codec,None)
        self.config=config;self.horizon=horizon;self.root=Path(config['log_dir'])
        self.weights=config['weights']
        assert 'interaction' not in self.weights, 'Rule-based reward does not support interaction dimension'
        assert set(self.weights.keys()) == {'position', 'persistence'}, 'Only position and persistence are supported'

    @torch.no_grad()
    def __call__(self,frames,context,actions,pixels,*,task,cursor,seed):
        begin=time.monotonic()
        codes=torch.stack([f['response'] for f in frames],1)
        if codes.shape[:2]!=(4,self.horizon):raise ValueError('Expected four horizon32 candidates')
        pred=self.worker.detokenize(TensorBatch(dict(tokens=codes,ctx_tokens=context.expand(4,-1,-1))),None).batch['pixels'][:,1:]
        def rgb(x):
            if not torch.isfinite(x).all():raise ValueError('Nonfinite decoded pixels')
            return x.clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()
        initial=rgb(pixels[0,:2]);reference=rgb(pixels[0,2:2+self.horizon])
        predictions=np.stack([rgb(x) for x in pred]);del pred
        scored=[rule_scores(reference,p) for p in predictions]
        axes={k:np.asarray([s[0][k] for s in scored]) for k in ('position','persistence')}
        
        reward=sum(float(self.weights[k])*axes[k] for k in self.weights)
        raw=reward.copy()
        if np.ptp(reward)<self.config.get('minimum_group_spread',.02):reward[:]=reward.mean()
        
        self.root.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(self.root/f'{cursor:06d}_clips.npz',initial=initial,reference=reference,predictions=predictions,tokens=codes.cpu().numpy())
        write_json(self.root/f'{cursor:06d}.json',dict(cursor=cursor,seed=seed,task=str(task),weights=self.weights,axes={k:v.tolist() for k,v in axes.items()},raw_reward=raw.tolist(),reward=reward.tolist(),tracks=[s[1] for s in scored],seconds=time.monotonic()-begin))
        
        stats=dict(judged_groups=1,informative_groups=int(np.ptp(reward)>1e-6),judge_seconds=time.monotonic()-begin,reward_spread=float(np.ptp(raw)),**{f'score_sum_{k}':float(v.sum()) for k,v in axes.items()})
        return torch.tensor(reward,dtype=torch.float32,device=codes.device),stats
