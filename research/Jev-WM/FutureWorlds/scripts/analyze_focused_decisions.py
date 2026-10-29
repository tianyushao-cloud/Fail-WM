"""Offline object diagnostics; never used as an RL reward."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
from futureworlds.focused_alignment import red_mask,crop_box

def component_centers(mask):
    labels,n=ndimage.label(mask)
    result=[]
    for label in range(1,n+1):
        yy,xx=np.where(labels==label)
        if len(xx)>=8:result.append((float(xx.mean()),float(yy.mean()),len(xx)))
    return result

def object_metrics(prediction,truth):
    pm,tm=red_mask(prediction),red_mask(truth)
    rows=[]
    for t,(p,g) in enumerate(zip(pm,tm)):
        pc,gc=component_centers(p),component_centers(g)
        separated=any(np.hypot(a[0]-b[0],a[1]-b[1])>=10 for i,a in enumerate(pc) for b in pc[i+1:])
        union=int((p|g).sum());inter=int((p&g).sum())
        rows.append(dict(frame=t+2,reference_red_pixels=int(g.sum()),prediction_red_pixels=int(p.sum()),
                         red_iou=inter/union if union else None,
                         missing_flag=bool(g.sum()>=8 and p.sum()<3),
                         duplicate_flag=bool(len(gc)==1 and len(pc)>=2 and separated)))
    valid=[r["red_iou"] for r in rows if r["red_iou"] is not None]
    return dict(red_iou=float(np.mean(valid)) if valid else None,
                missing_flags=sum(r["missing_flag"] for r in rows),duplicate_flags=sum(r["duplicate_flag"] for r in rows),
                per_frame=rows,limitations="Color/connected-component proxy only, not simulator task success. Occlusion and fragmentation can cause false positives; inspect images.")

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")

def comparison(root):
    root=Path(root)
    sft=json.loads((root/"evaluation/sft/summary.json").read_text())
    rl=json.loads((root/"evaluation/rl/summary.json").read_text())
    dest=root/"behavior";dest.mkdir(exist_ok=True)
    results=[]
    rng=np.random.default_rng(20261017)
    for i,(a,b) in enumerate(zip(sft["per_case"],rl["per_case"])):
        assert a["id"]==b["id"]
        with np.load(root/f"evaluation/sft/case_{i:06d}/prediction.npz") as x,np.load(root/f"evaluation/rl/case_{i:06d}/prediction.npz") as y:
            assert np.array_equal(x["truth"],y["truth"]) and np.array_equal(x["observations"],y["observations"])
            truth=x["truth"];initial=x["observations"]
            p=np.rint(np.clip(x["prediction"],0,1)*255).astype(np.uint8)
            q=np.rint(np.clip(y["prediction"],0,1)*255).astype(np.uint8)
            order=["sft","rl"] if rng.integers(2)==0 else ["rl","sft"]
            mapping=dict(zip(["A","B"],order))
            frames=dict(GT=truth,**{label:{"sft":p,"rl":q}[source] for label,source in mapping.items()})
            folder=dest/f"case_{i:06d}";folder.mkdir(exist_ok=True)
            box=crop_box(initial,truth)
            for part,start in enumerate([0,8,16,24]):
                canvas=Image.new("RGB",(1280,3*180),"white");draw=ImageDraw.Draw(canvas)
                for row,(label,sequence) in enumerate(frames.items()):
                    for col,t in enumerate(range(start,start+8)):
                        canvas.paste(Image.fromarray(sequence[t]).crop(box).resize((160,160)),(col*160,row*180+20))
                        draw.text((col*160,row*180),f"{label} t={t+2}",fill="black")
                canvas.save(folder/f"detail_{part}.jpg")
            write_json(folder/"label_key.json",mapping)
            results.append(dict(id=a["id"],sft=object_metrics(p,truth),rl=object_metrics(q,truth),
                                lpips_sft=a["metrics"]["32"]["lpips"],lpips_rl=b["metrics"]["32"]["lpips"]))
    report=dict(cases=results,aggregate={label:dict(
                red_iou=float(np.mean([r[label]["red_iou"] for r in results])),
                missing_flags=sum(r[label]["missing_flags"] for r in results),
                duplicate_flags=sum(r[label]["duplicate_flags"] for r in results)) for label in ("sft","rl")},
                caveat="Automated color proxies, not human labels or simulator success. Blinded A/B sheets cover all32 frames.")
    write_json(dest/"comparison.json",report)
    print(json.dumps(report["aggregate"]),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("root");args=parser.parse_args();comparison(args.root)

