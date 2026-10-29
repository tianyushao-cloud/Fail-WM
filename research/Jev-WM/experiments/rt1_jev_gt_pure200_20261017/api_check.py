"""CPU-only, bounded real API control check on fixed training windows."""
import json
from pathlib import Path
import sys
import numpy as np
from runtime import api_environment
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/src'))
from futureworlds.jev_judge import Store, consensus, atomic_json, sha


def main():
    api_environment()
    cfg=json.loads((ROOT/'train.json').read_text())
    dc={**cfg['decisions'],'backend':'api','review_dir':str(ROOT/'api_control_reviews_v2'),'max_http_attempts':32}
    store=Store(dc)
    manifest=Path(cfg['train_manifest'])
    rows=json.loads(manifest.read_text())['cases'][3:6]
    results=[]
    protocol=dict(rows=[r['id'] for r in rows], controls=['identical','GT_vs_black','GT_vs_freeze'],
                  gate='Every GT-vs-black comparison selects GT in both orders; identical has no preference; GT-vs-freeze must not prefer freeze. This is an interface/gross-error check, not proof of fine candidate ranking.')
    path=ROOT/'api_control_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Control protocol changed')
    atomic_json(path,protocol)
    for row in rows:
        file=manifest.parent/row['file']
        assert sha(file.read_bytes())==row['sha256']
        with np.load(file,allow_pickle=False) as z:
            initial=z['images'][:2].copy(); ref=z['images'][2:12].copy();task=str(z['instruction'].item())
        pred=np.stack([ref,ref,np.zeros_like(ref),np.repeat(initial[-1:],10,axis=0)])
        key,arrays,meta=store.group(initial,ref,pred,task)
        pairs=[]
        for name,b in [('identical',1),('GT_vs_black',2),('GT_vs_freeze',3)]:
            a,k1=store.request(key,arrays,meta,0,b)
            rev,k2=store.request(key,arrays,meta,b,0)
            result=dict(control=name,labels=[a,rev],vote=consensus(a,rev),requests=[k1,k2])
            pairs.append(result)
            print(json.dumps(dict(case=row['id'],**result)),flush=True)
        results.append(dict(id=row['id'],pairs=pairs))
        atomic_json(ROOT/'api_control_partial.json',dict(results=results))
    passed=all(r['pairs'][0]['vote']==0 and r['pairs'][1]['vote']==1 and r['pairs'][2]['vote']!=-1 for r in results)
    result=dict(state='PASS' if passed else 'FAILED',results=results,new_api_calls=store.new_calls,
                limitation='Synthetic controls; not an independent validation of semantic ranking accuracy.')
    atomic_json(ROOT/'api_control_check.json',result)
    print(json.dumps(dict(state=result['state'],new_api_calls=store.new_calls)),flush=True)
    if not passed: raise RuntimeError('Gross-error API control check failed; inspect recorded evidence')


if __name__=='__main__':
    main()
