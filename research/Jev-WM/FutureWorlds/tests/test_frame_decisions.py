import base64,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from futureworlds.frame_decisions import FrameStore,QUESTIONS,aggregate_frames
class FrameChecks(unittest.TestCase):
    def test_candidate_mapping_and_receipt_replay(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict("os.environ",{"OPENAI_API_KEY":"test-only"}):
            store=FrameStore(dict(model="gpt-6-luna",review_dir=tmp))
            images=np.stack([np.full((8,8,3),v,np.uint8) for v in [40,80,120,160]])
            def answer(payload):
                # Read the labeled FULL pixels, so reversed candidates must remap correctly.
                parts=payload["input"][0]["content"];values={}
                for i,p in enumerate(parts):
                    if p["type"]=="input_text" and "FULL view" in p["text"]:
                        label=p["text"].split(";")[0]
                        pixels=np.array(Image.open(io.BytesIO(base64.b64decode(parts[i+1]["image_url"].split(",")[1]))))
                        values[label]=int(pixels[0,0,0])//40-1
                answers=[]
                for q in payload["questions"]:
                    label=q["name"][0];top=len(q["levels"])-1;value=min(top,values[label])
                    answers.append(dict(name=q["name"],type="score",score=value,confidence=1,
                                        probabilities=[dict(value=i,label=l["label"],probability=float(i==value))
                                                       for i,l in enumerate(q["levels"])]))
                return dict(model="gpt-6-luna",answers=answers),dict(test_only=True)
            with patch.object(store.client,"score",side_effect=answer) as call:
                a=store.frame(images[:2],images,np.zeros(52),images[0],"cube",8,[0,0,8,8],False)
                b=store.frame(images[:2],images,np.zeros(52),images[0],"cube",8,[0,0,8,8],True)
                self.assertNotEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
                self.assertEqual(call.call_count,2)
                store.frame(images[:2],images,np.zeros(52),images[0],"cube",8,[0,0,8,8],True)
                self.assertEqual(call.call_count,2)
                receipt=Path(tmp)/a[0]/"score.json";v=json.loads(receipt.read_text());v["request_id"]="other";receipt.write_text(json.dumps(v))
                with self.assertRaisesRegex(ValueError,"mismatch"):
                    store.frame(images[:2],images,np.zeros(52),images[0],"cube",8,[0,0,8,8],False)
    def test_rare_object_error_and_black_are_penalized(self):
        values=np.ones((8,4,3))
        values[2,1,1]=.25
        values[2,2,2]=0
        values[:,3,0]=.25
        result=aggregate_frames(values)
        self.assertEqual(result[0],1)
        self.assertLess(result[1],.8)
        self.assertEqual(result[2],0)
        self.assertEqual(result[3],.5)
if __name__=="__main__":unittest.main()

