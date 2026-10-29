"""Re-execute real native actions to resample correctly synchronized RGB/state.

Raw actions are byte-identical. Require every qpos, qvel and target13 to match
original before accepting. No altered motion, object teleportation or video edit.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
import argparse,json,hashlib,time,shutil,multiprocessing as mp
from pathlib import Path

def worker(args):
    src,dst=args;src=Path(src);dst=Path(dst)
    import mujoco,h5py,numpy as np
    from rc_collect.environment import make_env
    from rc_collect.recording import Recorder
    from rc_collect.batch_validation import validate_episode
    from rc_collect.synchronization import synchronize_observation
    old=json.loads((src/'metadata.json').read_text());spec=old.get('initial_spec',{})
    env=make_env(seed=old.get('seed',20261003),initial_spec=spec)
    with h5py.File(src/'trajectory.h5') as f:
        raw=f['actions/raw'][:];original_q=f['observations/qpos'][:];original_v=f['observations/qvel'][:];original_target=f['actions/target13'][:];phases=f['actions/phase'].asstr()[:];state_spec=int(f.attrs['mujoco_integration_state_spec']);initial=f['simulator/integration_state'][0]
        mujoco.mj_setState(env.sim.model._model,env.sim.data._data,initial,state_spec);env.sim.forward()
        env.robots[0].gripper['right'].current_action=f['simulator/initial_gripper_action'][:]
        for c in env.robots[0].part_controllers.values():c.update(force=True);c.reset_goal()
        mujoco.mj_setState(env.sim.model._model,env.sim.data._data,initial,state_spec);synchronize_observation(env)
        keys=['episode_id','initial_state_id','seed','action_seed','split','category','group_index','branch_index','reference_profile','initial_spec','collection_plan_hash','recovery_attempted','controller_phase_timeouts']
        info={k:old[k] for k in keys if k in old};info.update(schema_version='2.1',action_generation_version='2.0_cached_feedback',observation_resampled_from_native_replay=True,raw_action_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
        tmp=dst.parent.parent/'sync_incomplete'/src.name;tmp.mkdir(exist_ok=True,parents=True)
        rec=Recorder(env,tmp,old['branch'],episode_info=info)
        qerr=float(np.max(abs(env.sim.data.qpos-original_q[0])));verr=float(np.max(abs(env.sim.data.qvel-original_v[0])))
        for i,(a,p) in enumerate(zip(raw,phases)):
            rec.step(a,p);qerr=max(qerr,float(np.max(abs(env.sim.data.qpos-original_q[i+1]))));verr=max(verr,float(np.max(abs(env.sim.data.qvel-original_v[i+1]))))
        meta=rec.finish(old['perturbation']);terr=float(np.max(abs(np.asarray(rec.targets)-original_target)))
        assert np.asarray(rec.raw).tobytes()==raw.tobytes(),'native actions changed'
        assert qerr<1e-6 and verr<1e-5 and terr<1e-6,(qerr,verr,terr)
        meta['success_changed_by_state_sync']=meta['success']!=old['success']
        meta['sampling_fix_original_replay_errors']={'qpos':qerr,'qvel':verr,'target13':terr,'raw_actions_byte_identical':True}
        validation=validate_episode(tmp,env);meta['validation']=validation;meta['complete']=True
        for k in ['physical_parameters','wall_seconds']:
            if k in old:meta[k]=old[k]
        (tmp/'metadata.json').write_text(json.dumps(meta,indent=2));(tmp/'validation.json').write_text(json.dumps(validation,indent=2))
        env.close();dst.parent.mkdir(exist_ok=True,parents=True)
        assert not dst.exists(),dst
        tmp.replace(dst)
        return {'episode_id':meta['episode_id'],'success':meta['success'],'failure_type':meta['failure_type'],'errors':meta['sampling_fix_original_replay_errors']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--destination',required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--preview-source');ap.add_argument('--preview-destination');a=ap.parse_args()
    src=Path(a.source).resolve();dst=Path(a.destination).resolve();jobs=[(str(p),str(dst/p.name)) for p in sorted(src.iterdir()) if (p/'metadata.json').exists() and not (dst/p.name/'metadata.json').exists()]
    if a.limit:jobs=jobs[:a.limit]
    if a.preview_source:
        ps=Path(a.preview_source).resolve();pd=Path(a.preview_destination).resolve()
        jobs=[(str(p),str(pd/p.name)) for p in sorted(ps.iterdir()) if (p/'metadata.json').exists() and not (pd/p.name/'metadata.json').exists()]+jobs
    results=[];start=time.time()
    with mp.get_context('spawn').Pool(a.workers,maxtasksperchild=25) as pool:
        for result in pool.imap_unordered(worker,jobs,chunksize=1):
            results.append(result);report={'completed_this_run':len(results),'requested_this_run':len(jobs),'wall_seconds':round(time.time()-start,2),'last':result}
            (dst.parent/'sampling_fix_progress.json').write_text(json.dumps(report,indent=2))
            if len(results)%10==0:print(json.dumps(report),flush=True)
    (dst.parent/'sampling_fix_report.json').write_text(json.dumps(results,indent=2));print('COMPLETE',len(results),flush=True)
if __name__=='__main__':main()
