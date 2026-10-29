"""Shared-filesystem API queue: GPU workers never need credentials or API egress."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
from runtime import api_environment
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/src'))
from futureworlds.jev_judge import Store,atomic_json,sha,canonical


def main():
    api_environment()
    cfg=json.loads((ROOT/'train.json').read_text())['decisions']
    cfg={**cfg,'backend':'api'}
    root=Path(cfg['review_dir']);root.mkdir(exist_ok=True)
    lock=(root/'.broker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def work(path):
        identity=json.loads(path.read_text());key=identity['group']
        if len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid group key')
        folder=root/'groups'/key
        meta=json.loads((folder/'request.json').read_text())
        if sha(canonical(meta).encode())!=key:raise ValueError('Group metadata mismatch')
        with np.load(folder/'frames.npz',allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in meta['arrays']}
        for k,v in arrays.items():
            if v.dtype!=np.uint8 or list(v.shape)!=meta['arrays'][k]['shape'] or sha(v.tobytes())!=meta['arrays'][k]['sha256']:
                raise ValueError('Group arrays corrupted')
        Store(cfg).request(key,arrays,meta,identity['A'],identity['B'])
    active={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        while not stopped:
            atomic_json(root/'broker.json',dict(pid=os.getpid(),heartbeat_unix=time.time(),active=len(active),state='RUNNING'))
            for key,(future,path) in list(active.items()):
                if future.done():
                    try:future.result()
                    except Exception as error:
                        atomic_json(root/'errors'/path.name,dict(error=f'{type(error).__name__}: {error}',time=time.time()))
                        print(json.dumps(dict(request=key,error=type(error).__name__)),flush=True)
                    path.unlink(missing_ok=True)
                    del active[key]
            for path in sorted((root/'queue').glob('*.json')):
                if len(active)>=4:break
                if path.stem not in active:
                    active[path.stem]=(pool.submit(work,path),path)
            time.sleep(.5)
    atomic_json(root/'broker.json',dict(pid=os.getpid(),heartbeat_unix=time.time(),state='STOPPED'))


if __name__=='__main__':main()
