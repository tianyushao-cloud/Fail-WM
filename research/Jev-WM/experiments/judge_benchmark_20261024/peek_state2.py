import json,h5py,numpy as np,collections
man={r["id"]:r for r in json.load(open("/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/train_natural.json"))["cases"]}
S={};C=collections.defaultdict(list)
for l in open("results/state.jsonl"):
    r=json.loads(l);f=r["key"].split("|")
    if f[0]!="gt" or f[1]!="raw": continue
    cid=f[2];t=int(f[3])
    if cid not in S:
        with h5py.File(man[cid]["source_h5"]) as h:
            i=(np.arange(32)+2)*4; S[cid]=dict(fl=h["events/flags"][:][i], dz=(h["observations/object_pose_wxyz"][:,2][i]-h["observations/object_pose_wxyz"][0,2])*100)
    fl=S[cid]["fl"][t]; dz=S[cid]["dz"][t]
    if fl[2]: continue
    phase = "grasped" if fl[1] else ("contact" if fl[0] else "free")
    C[(man[cid]["failure_type"] or "success", phase)].append(r["p"][1]>.5)
print("GT NOT lifted frames: Qwen says lifted, by failure type x phase")
for k in sorted(C): print("  %-24s %-8s n=%4d  false-lift %.2f"%(k[0],k[1],len(C[k]),np.mean(C[k])))
