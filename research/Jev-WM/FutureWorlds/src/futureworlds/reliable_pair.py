"""Conservative pure-Decisions preferences: learn one corroborated pair, not a full rank."""
import json
import numpy as np
import torch
from .focused_alignment import FocusedStore, aggregate, rgb
from .batch import TensorBatch
from .rewards import CodecWorker
from .decision_reward import canonical, write_json
import hashlib

POLICY = dict(minimum_score_gap=.04, minimum_object_gap=.04,
              maximum_component_order_delta=.15, minimum_winner_object=.50)
COMPONENTS = ("fidelity", "object_behavior", "usable_scene")

def select_pair(components):
    """Two orders x candidates x three components. Others have zero PG advantage.

    Thresholds are fixed before independent calibration. There is no metric reward,
    no forced ranking and no preference for candidate index. Exact best-pair ties
    abstain rather than using screen position to break a tie.
    """
    x=np.asarray(components,dtype=np.float64)
    if x.ndim!=3 or x.shape[0]!=2 or x.shape[2]!=3 or x.shape[1]<2:
        raise ValueError("Expected two orders of candidate components")
    if not np.isfinite(x).all() or (x<0).any() or (x>1).any():
        raise ValueError("Components must be finite normalized scores")
    score=np.sqrt(x[:,:,0]*x[:,:,1])*x[:,:,2]
    delta=np.abs(x[0]-x[1]).max(axis=1)
    eligible=[]
    for win in range(x.shape[1]):
        for lose in range(x.shape[1]):
            if win==lose:continue
            gap=x[:,win]-x[:,lose]
            margin=float((score[:,win]-score[:,lose]).min())
            if (max(delta[win],delta[lose])<=POLICY["maximum_component_order_delta"]
                and margin>=POLICY["minimum_score_gap"]
                and gap[:,1].min()>=POLICY["minimum_object_gap"]
                and gap[:,0].min()>=0 and gap[:,2].min()>=0
                and x[:,win,1].min()>=POLICY["minimum_winner_object"]):
                eligible.append(dict(winner=win,loser=lose,minimum_gap=margin,
                                     component_gaps=gap.tolist()))
    eligible.sort(key=lambda p:p["minimum_gap"],reverse=True)
    rewards=np.zeros(x.shape[1])
    chosen=None
    if eligible and (len(eligible)==1 or eligible[0]["minimum_gap"]-eligible[1]["minimum_gap"]>1e-8):
        chosen=eligible[0]
        rewards[chosen["winner"]]=1.
        rewards[chosen["loser"]]=-1.
    return rewards,dict(policy=POLICY,eligible_pairs=eligible,chosen=chosen,
                        rejected=chosen is None,order_scores=score.tolist(),
                        component_order_delta=delta.tolist())

class ReliablePairReward:
    def __init__(self,codec,config,dataset):
        if dataset!="robocasa":raise ValueError("RoboCasa calibration only")
        self.worker=CodecWorker(codec,None);self.store=FocusedStore(config);self.dataset=dataset
    @torch.no_grad()
    def __call__(self,frames,context,actions,pixels,*,task,raw_actions):
        codes=torch.stack([f["response"] for f in frames],1)
        if codes.min()<0 or codes.max()>=4375:raise ValueError("Invalid visual tokens")
        pred=self.worker.detokenize(TensorBatch(dict(tokens=codes,ctx_tokens=context.expand(len(codes),-1,-1))),None).batch["pixels"][:,1:]
        initial,reference=rgb(pixels[0,:2]),rgb(pixels[0,2:2+len(frames)])
        components=[];keys=[]
        for reverse in (False,True):
            self.store.prediction_first=reverse
            rows=[];orderkeys=[]
            for p in pred:
                key,_=self.store.score_arrays(initial,rgb(p),raw_actions[2:2+len(frames)],task,self.dataset,reference)
                s=self.store.components(key);rows.append([s[n] for n in COMPONENTS]);orderkeys.append(key)
            components.append(rows);keys.append(orderkeys)
        values,stats=select_pair(components)
        evidence=dict(keys=keys,components=components,rewards=values.tolist(),selection=stats)
        digest=hashlib.sha256(canonical(evidence).encode()).hexdigest()
        write_json(self.store.root/"pairs"/(digest+".json"),evidence)
        rewards=torch.tensor(values,device=codes.device,dtype=torch.float32)
        return rewards,{"decisions/mean":float(rewards.mean()),"decisions/std":float(rewards.std(unbiased=False)),
                        "decisions/all_tied":float(stats["rejected"]),
                        "decisions/eligible_pairs":float(len(stats["eligible_pairs"])),
                        "decisions/selected_margin":float(stats["chosen"]["minimum_gap"] if stats["chosen"] else 0),
                        "decisions/api_calls":float(self.store.client.calls if self.store.client else 0)}

