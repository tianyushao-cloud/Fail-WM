"""Verify every recorded transition; open-loop replay on an independent reset."""
import json
import h5py,numpy as np,mujoco,imageio.v2 as imageio
from .action_conversion import controller_target13,executed13
from .synchronization import synchronize_observation

def validate_episode(path,env=None):
    with h5py.File(path/'trajectory.h5','r') as f:
        a=f['actions/raw'][:];q=f['observations/qpos'][:];v=f['observations/qvel'][:];ts=f['observations/timestamp'][:];target=f['actions/target13'][:];T=len(a)
        assert a.shape==(T,12) and target.shape==(T,13)
        for key in ['observations','events']:
            assert all(len(d)==T+1 for d in f[key].values()),key
        assert all(len(d)==T for d in f['actions'].values())
        assert np.allclose(np.diff(ts),.05,atol=1e-9,rtol=0)
        assert np.allclose(f['actions/timestamp'][:],ts[:-1],atol=1e-9,rtol=0)
        assert np.array_equal(f['observations/timestep'][:],np.arange(T+1))
        assert np.isfinite(a).all() and np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(target).all()
        R=target[:,3:12].reshape(-1,3,3)
        assert np.max(abs(R@R.transpose(0,2,1)-np.eye(3)))<1e-6 and np.max(abs(np.linalg.det(R)-1))<1e-6
        assert np.all(a[:,7:11]==0) and np.all(a[:,11]==-1)
        reader=imageio.get_reader(str(path/'rgb.mp4'));video_meta=reader.get_meta_data();frames=sum(1 for _ in reader);reader.close()
        assert frames==T+1 and abs(video_meta['fps']-20)<1e-9
        assert abs(video_meta['duration']-(T+1)/20)<.011
        report={'alignment_pass':True,'video_frames':frames,'fps':20,'action_duration_seconds':T/20,'video_duration_seconds':video_meta['duration']}
        if env is not None:
            # Fresh controllers and scene (given same spec/seed), then restore full integration state.
            env.reset();from .initial_state import prepare_initial_state
            prepare_initial_state(env)
            spec=int(f.attrs['mujoco_integration_state_spec']);init=f['simulator/integration_state'][0]
            mujoco.mj_setState(env.sim.model._model,env.sim.data._data,init,spec);env.sim.forward()
            env.robots[0].gripper['right'].current_action=f['simulator/initial_gripper_action'][:]
            for c in env.robots[0].part_controllers.values():c.update(force=True);c.reset_goal()
            mujoco.mj_setState(env.sim.model._model,env.sim.data._data,init,spec)
            synchronize_observation(env)
            original=env._pre_action;captured=[]
            def hook(action,policy_step=False):
                original(action,policy_step=policy_step)
                if policy_step:captured.append(controller_target13(env).copy())
            env._pre_action=hook
            qerr=0.;verr=0.;poseerr=0.;objecterr=0.;rgb_max=0;rgb_mae=0.;samples=set(np.linspace(0,T,7,dtype=int))
            try:
                for i in range(T+1):
                    if i:env.step(a[i-1])
                    synchronize_observation(env)
                    poseerr=max(poseerr,float(np.max(abs(executed13(env)-f["observations/eef_executed13"][i]))))
                    objecterr=max(objecterr,float(np.max(abs(np.r_[env.sim.data.body_xpos[env.obj_body_id["cube"]],env.sim.data.body_xquat[env.obj_body_id["cube"]]]-f["observations/object_pose_wxyz"][i]))))
                    qerr=max(qerr,float(np.max(abs(env.sim.data.qpos-q[i]))));verr=max(verr,float(np.max(abs(env.sim.data.qvel-v[i]))))
                    if i in samples:
                        im=env.sim.render(640,480,camera_name='preview')[::-1];diff=abs(im.astype(np.float32)-f['observations/rgb'][i].astype(np.float32))
                        rgb_max=max(rgb_max,int(diff.max()));rgb_mae=max(rgb_mae,float(diff.mean()))
            finally:env._pre_action=original
            terr=float(np.max(abs(np.asarray(captured)-target)));actual_success=bool(env._check_success());expected_success=bool(f['events/flags'][-1,-1])
            ok=poseerr<1e-6 and objecterr<1e-6 and qerr<1e-6 and verr<1e-5 and terr<1e-6 and rgb_mae<.05 and actual_success==expected_success
            report.update(sampling_sync_pass=True,replay_max_executed13_error=poseerr,replay_max_object_pose_error=objecterr,replay_pass=ok,replay_max_qpos_error=qerr,replay_max_qvel_error=verr,replay_max_target13_error=terr,replay_sampled_rgb_max_error=rgb_max,replay_sampled_rgb_max_mae=rgb_mae,replay_rgb_frames_checked=len(samples),replay_success_matches=actual_success==expected_success)
            if not ok:raise RuntimeError('replay mismatch: '+json.dumps(report))
        return report
