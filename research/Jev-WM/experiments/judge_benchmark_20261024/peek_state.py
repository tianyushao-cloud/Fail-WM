import json,h5py,numpy as np
man={r["id"]:r for r in json.load(open("/media/tyshao/fail_WM/FutureWorlds_SFT/robocasa_data/train_natural.json"))["cases"]}
L={};y=[];p=[];src=[];n=0
for l in open("results/state.jsonl"):
    r=json.loads(l);n+=1;f=r["key"].split("|")
    if f[0]!="gt": continue
    cid=f[2]
    if cid not in L:
        with h5py.File(man[cid]["source_h5"]) as h: L[cid]=h["events/flags"][:,2][(np.arange(32)+2)*4]
    y.append(L[cid][int(f[3])]); p.append(r["p"]); src.append(f[1])
y=np.array(y);p=np.array(p);src=np.array(src)
print("done calls",n)
for s in ("raw","recon"):
    m=src==s
    if m.sum()==0: continue
    pl=p[m][:,1]>.5
    print(s,"n",m.sum(),"acc %.3f lifted-recall %.3f table-recall %.3f mean P(missing) %.3f"%((pl==y[m]).mean(),pl[y[m]==1].mean(),(~pl[y[m]==0]).mean(),p[m][:,2].mean()))
