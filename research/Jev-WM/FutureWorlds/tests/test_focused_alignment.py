import base64,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from futureworlds.focused_alignment import FocusedStore,FocusedReward,QUESTIONS,aggregate,crop_box,order_consensus
from futureworlds.batch import TensorBatch
from futureworlds.decision_reward import write_json

class FocusedChecks(unittest.TestCase):
    def test_payload_reference_crop_order_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=FocusedStore(dict(backend="replay",model="gpt-6-luna",review_dir=tmp))
            ref=np.zeros((32,256,320,3),dtype=np.uint8);ref[:,160:170,210:220,0]=200
            pred=ref.copy();pred[:,5,5]=255
            args=(ref[:2],pred,np.zeros((32,52)),"cube","robocasa",ref)
            key=store.prepare(*args)
            packet=json.loads((Path(tmp)/key/"request.json").read_text())
            with np.load(Path(tmp)/key/"frames.npz") as arrays:payload=store.build_payload(packet,arrays)
            images=[c for c in payload["input"][0]["content"] if c["type"]=="input_image"]
            self.assertEqual(len(images),48)
            decode=lambda p:np.array(Image.open(io.BytesIO(base64.b64decode(p["image_url"].split(",")[1]))))
            np.testing.assert_array_equal(decode(images[0]),ref[0])
            np.testing.assert_array_equal(decode(images[1]),pred[0])
            store.prediction_first=True
            reverse=store.prepare(*args)
            self.assertNotEqual(key,reverse)
            packet2=json.loads((Path(tmp)/reverse/"request.json").read_text())
            self.assertEqual(packet["crop_box"],packet2["crop_box"])
            with np.load(Path(tmp)/reverse/"frames.npz") as arrays:payload2=store.build_payload(packet2,arrays)
            images2=[c for c in payload2["input"][0]["content"] if c["type"]=="input_image"]
            np.testing.assert_array_equal(decode(images2[0]),pred[0])
            changed=pred.copy();changed[:,:,0]=255
            self.assertEqual(crop_box(ref[:2],ref),packet["crop_box"])
            self.assertNotEqual(key,store.prepare(ref[:2],changed,args[2],"cube","robocasa",ref))

    def test_aggregate_no_background_compensation(self):
        self.assertEqual(aggregate(dict(fidelity=1,object_behavior=0,usable_scene=1)),0)
        self.assertEqual(aggregate(dict(fidelity=1,object_behavior=1,usable_scene=0)),0)
        self.assertAlmostEqual(aggregate(dict(fidelity=.81,object_behavior=.25,usable_scene=1)),.45)

    def test_official_receipt_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=FocusedStore(dict(backend="replay",model="gpt-6-luna",review_dir=tmp))
            a=np.zeros((2,8,8,3),np.uint8)
            key=store.prepare(a,a,np.zeros((2,52)),"cube","robocasa",a)
            answers=[dict(name=q["name"],type="score",score=len(q["levels"])-1,confidence=1,
                     probabilities=[dict(value=i,label=l["label"],probability=float(i==len(q["levels"])-1))
                                    for i,l in enumerate(q["levels"])]) for q in QUESTIONS]
            receipt=dict(schema_version=1,request_id=key,model=store.model,provenance="synthetic unit test",
                         scoring_protocol="openai_decisions_images_v1",official_response=dict(model=store.model,answers=answers))
            write_json(store.root/key/"score.json",receipt)
            self.assertEqual(store.read(key),1)
            receipt["provenance"]="changed"
            write_json(store.root/key/"score.json",receipt)
            with self.assertRaisesRegex(ValueError,"changed"):store.read(key)

    def test_tiny_reversal_ties_ambiguous_pair_without_changing_best(self):
        values,stats=order_consensus([[.481,.408,.508,.455],[.454,.412,.506,.404]])
        self.assertFalse(stats["rejected"])
        self.assertEqual(values[1],values[3])
        self.assertEqual(int(values.argmax()),2)
        self.assertEqual(stats["ambiguous_pairs"],[[1,3]])

    def test_order_reversal_suppresses_policy_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            reward=FocusedReward(None,dict(backend="replay",model="gpt-6-luna",review_dir=tmp),"robocasa")
            reward.worker.detokenize=lambda *a:TensorBatch(dict(pixels=torch.zeros(4,3,3,8,8)))
            vals=iter([.1,.3,.5,.8,.8,.5,.3,.1])
            reward.store.score_arrays=lambda *a:("fake-test",next(vals))
            frames=[dict(response=torch.zeros(4,80,dtype=torch.long)) for _ in range(2)]
            rewards,stats=reward(frames,torch.zeros(1,1,1),None,torch.zeros(1,4,3,8,8),task="cube",raw_actions=np.zeros((4,52)))
            self.assertEqual(float(rewards.std()),0)
            self.assertEqual(stats["decisions/order_reversal"],1)
            self.assertEqual(stats["decisions/all_tied"],1)
if __name__=="__main__":unittest.main()

