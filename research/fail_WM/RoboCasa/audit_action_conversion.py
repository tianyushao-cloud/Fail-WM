"""Independent native OSC-delta -> target13 algebra, with current state pose."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from scipy.spatial.transform import Rotation

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/batch1000');a=ap.parse_args();root=Path(a.data);folders=root/'episodes' if (root/'episodes').exists() else root;reports=[]
    for p in sorted(folders.iterdir()):
        if not (p/'trajectory.h5').exists():continue
        with h5py.File(p/'trajectory.h5') as f:
            raw=f['actions/raw'][:];cmd=f['actions/target13'][:];state=f['observations/eef_executed13'][:-1]
            predicted=np.empty_like(cmd);predicted[:,:3]=state[:,:3]+.04*raw[:,:3]
            R=Rotation.from_rotvec(.25*raw[:,3:6]).as_matrix()@state[:,3:12].reshape(-1,3,3);predicted[:,3:12]=R.reshape(-1,9)
            g=float(f['simulator/initial_gripper_action'][0])
            for i in range(len(raw)):g=float(np.clip(g-.2*np.sign(raw[i,6]),-1,1));predicted[i,12]=g
            err=float(np.max(abs(predicted-cmd)));assert err<1e-6,(p,err)
            reports.append({'episode':p.name,'max_native_to_target13_error':err,'pass':True})
    out={'episodes':len(reports),'all_pass':all(r['pass'] for r in reports),'max_error':max(r['max_native_to_target13_error'] for r in reports),'results':reports}
    (root/'action_conversion_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='results'}))
if __name__=='__main__':main()
