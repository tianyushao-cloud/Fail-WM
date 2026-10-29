"""Synthetic unit/distributed tests only; never export test scores to training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/src'))
from futureworlds.jev_judge import classify,consensus,pairing,Store,canonical,sha,atomic_json


def response(label,probs=None):
    ps=probs or {x:float(x==label) for x in 'ABCD'}
    return dict(model='gpt-6-luna',answers=[dict(name='reference_fidelity',type='choice',choice=label,confidence=.9,
        probabilities=[dict(value=k,probability=v) for k,v in ps.items()])])


def distributed_worker(rank,path):
    import torch.distributed as dist
    from datetime import timedelta
    from futureworlds.jev_gt_reward import synchronize_failure
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=4,timeout=timedelta(seconds=40))
    synchronize_failure(None,4)
    try:synchronize_failure(ValueError('TEST_ONLY_failure') if rank==2 else None,4)
    except RuntimeError as e:assert 'TEST_ONLY_failure' in str(e)
    else:raise AssertionError('All ranks must fail before backward')
    dist.destroy_process_group()


class Tests(unittest.TestCase):
    def test_labels_not_ordinal_rewards(self):
        for label in 'ABCD':self.assertEqual(classify(response(label),'gpt-6-luna'),label)
        self.assertEqual(consensus('A','B'),1);self.assertEqual(consensus('B','A'),-1)
        for a,b in [('A','A'),('B','B'),('C','C'),('D','D'),('A','D'),('C','B')]:self.assertEqual(consensus(a,b),0)
    def test_ambiguous_and_invalid(self):
        self.assertEqual(classify(response('A',dict(A=.5,B=.4,C=.1,D=0)),'gpt-6-luna'),'D')
        for bad in [dict(A=float('nan'),B=0,C=0,D=0),dict(A=1,B=1,C=0,D=0)]:
            with self.assertRaises(ValueError):classify(response('A',bad),'gpt-6-luna')
        r=response('A');r['answers'][0]['type']='refusal'
        with self.assertRaises(ValueError):classify(r,'gpt-6-luna')
    def test_balanced_metric_free_pairs(self):
        pairs=set()
        for cursor in range(80):
            p=pairing(20260915,cursor);self.assertEqual(sorted(sum(p,[])),[0,1,2,3])
            self.assertEqual(p,pairing(20260915,cursor));pairs.update(tuple(sorted(x)) for x in p)
        self.assertEqual(len(pairs),6)
    def test_zero_advantage_for_abstained_pair(self):
        import torch
        from futureworlds.objective import compute_grpo_outcome_advantage
        rewards=torch.zeros(4,800);rewards[:,-1]=torch.tensor([1.,-1.,0.,0.])
        adv,_=compute_grpo_outcome_advantage(rewards,torch.ones_like(rewards),np.zeros(4,dtype=int))
        self.assertTrue(torch.all(adv[2:]==0));self.assertTrue(torch.all(adv[0]>0));self.assertTrue(torch.all(adv[1]<0))
    def test_group_binds_gt_and_task(self):
        with tempfile.TemporaryDirectory() as root:
            s=Store(dict(review_dir=root,model='gpt-6-luna',backend='replay'))
            initial=np.zeros((2,256,320,3),np.uint8);gt=np.zeros((10,256,320,3),np.uint8);pred=np.zeros((4,10,256,320,3),np.uint8)
            a,_,_=s.group(initial,gt,pred,'task');gt[0,0,0,0]=1
            b,_,_=s.group(initial,gt,pred,'task');c,_,_=s.group(initial,gt,pred,'other')
            self.assertNotEqual(a,b);self.assertNotEqual(b,c)
    def test_queue_delivery_and_immutable_receipt(self):
        import threading,time
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            store=Store(dict(review_dir=str(root),model='gpt-6-luna',backend='queue'))
            initial=np.zeros((2,256,320,3),np.uint8);gt=np.zeros((10,256,320,3),np.uint8);pred=np.zeros((4,10,256,320,3),np.uint8)
            key,arrays,meta=store.group(initial,gt,pred,'TEST_ONLY')
            atomic_json(root/'broker.json',dict(heartbeat_unix=time.time(),state='RUNNING'))
            def fake_worker():
                deadline=time.monotonic()+10
                while time.monotonic()<deadline:
                    entries=list((root/'queue').glob('*.json'))
                    if entries:
                        path=entries[0];identity=json.loads(path.read_text())
                        atomic_json(root/'requests'/path.stem/'receipt.json',dict(identity=identity,response=response('A'),provenance='TEST_ONLY'))
                        return
                    time.sleep(.05)
            thread=threading.Thread(target=fake_worker);thread.start()
            label,rid=store.request(key,arrays,meta,0,1);thread.join()
            self.assertEqual(label,'A')
            path=root/'requests'/rid/'receipt.json';receipt=json.loads(path.read_text());receipt['response']=response('B');atomic_json(path,receipt)
            with self.assertRaises(ValueError):store.request(key,arrays,meta,0,1)
    def test_four_rank_failure_propagation(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(distributed_worker,args=(root+'/rendezvous',),nprocs=4,join=True)


if __name__=='__main__':unittest.main()
