import base64
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from futureworlds.reference_alignment import ReferenceAlignmentStore, sampled_indices, QUESTIONS
from futureworlds.decision_api import DecisionsClient

class ReferenceAlignmentChecks(unittest.TestCase):
    def test_reference_identity_and_synchronized_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ReferenceAlignmentStore(dict(backend="replay",model="gpt-6-luna",review_dir=tmp))
            ref=np.stack([np.full((4,5,3),i,np.uint8) for i in range(32)])
            pred=ref+100
            initial=np.zeros((2,4,5,3),np.uint8)
            actions=np.zeros((32,52),np.float32)
            key=store.prepare(initial,pred,actions,"move","robocasa",ref)
            changed=ref.copy();changed[0]+=1
            self.assertNotEqual(key,store.prepare(initial,pred,actions,"move","robocasa",changed))
            packet=json.loads((Path(tmp)/key/"request.json").read_text())
            with np.load(Path(tmp)/key/"frames.npz") as arrays:
                payload=store.build_payload(packet,arrays)
            images=[p for p in payload["input"][0]["content"] if p["type"]=="input_image"]
            self.assertEqual(len(images),8)
            self.assertEqual(sampled_indices(32)[-1],31)
            for image,offset in zip(images,sampled_indices(32)):
                pixels=np.array(Image.open(io.BytesIO(base64.b64decode(image["image_url"].split(",")[1]))))
                np.testing.assert_array_equal(pixels[:,:5],ref[offset])
                np.testing.assert_array_equal(pixels[:,5:],pred[offset])
            with self.assertRaises(ValueError):
                store.prepare(initial,pred,actions,"move","robocasa",ref[:-1])

    def test_reward_uses_aligned_reference_and_does_not_amplify_tiny_range(self):
        import torch
        from futureworlds.batch import TensorBatch
        from futureworlds.reference_alignment import ReferenceAlignmentReward
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(backend="replay",model="gpt-6-luna",review_dir=tmp,minimum_range=.03)
            reward=ReferenceAlignmentReward(None,cfg,"robocasa")
            decoded=torch.zeros(4,3,3,4,5)
            reward.worker.detokenize=lambda *args:TensorBatch({"pixels":decoded})
            pixels=torch.stack([torch.full((3,4,5),i/10) for i in range(4)])[None]
            frames=[{"response":torch.zeros(4,80,dtype=torch.long)} for _ in range(2)]
            actions=np.arange(4*52,dtype=np.float32).reshape(4,52)
            seen=[]
            def score(initial,pred,raw,task,dataset,reference):
                np.testing.assert_array_equal(reference[0],np.full((4,5,3),51,np.uint8))
                np.testing.assert_array_equal(raw,actions[2:])
                seen.append(True)
                return str(len(seen)),.5+len(seen)*.001
            reward.store.score_arrays=score
            result,stats=reward(frames,torch.zeros(1,1,1),None,pixels,task="move",raw_actions=actions)
            self.assertEqual(len(seen),4)
            self.assertEqual(float(result.max()-result.min()),0)
            self.assertEqual(stats["decisions/all_tied"],1.0)
            self.assertGreater(stats["decisions/raw_range"],0)

    def test_retry_after_stops_without_retry_or_secret(self):
        with patch.dict("os.environ",{"OPENAI_API_KEY":"test-secret"}):
            client=DecisionsClient(max_calls=3,retries=2)
            err=urllib.error.HTTPError("https://api.openai.com/v1/decisions",429,"limited",
                                      {"Retry-After":"1728"},io.BytesIO(b"test-secret"))
            with patch.object(client.opener,"open",side_effect=err) as send,patch("time.sleep"):
                with self.assertRaisesRegex(RuntimeError,"Retry-After=1728") as caught:
                    client.score(dict(model="gpt-6-luna",questions=QUESTIONS,input="test"))
                self.assertEqual(send.call_count,1)
                self.assertNotIn("test-secret",str(caught.exception))
                self.assertEqual(client.last_error["status"],429)

if __name__=="__main__":unittest.main()

