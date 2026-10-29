from pathlib import Path
import json,hashlib
import numpy as np
import h5py,imageio.v2 as imageio,mujoco
from .events import EventLabeler,EVENT_NAMES
from .action_conversion import executed13,controller_target13
from .synchronization import synchronize_observation

class Recorder:
    def __init__(self,env,path,kind,width=640,height=480,episode_info=None):
        self.env=env;self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.kind=kind
        self.episode_info=dict(episode_info or {})
        self.w=width;self.h=height;self.labeler=EventLabeler(env);self.data={};self.raw=[];self.targets=[];self.phases=[];self.times=[]
        self.writer=imageio.get_writer(str(self.path/'rgb.mp4'),fps=20,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-threads','1'])
        self.spec=int(mujoco.mjtState.mjSTATE_INTEGRATION);self.nstate=mujoco.mj_stateSize(env.sim.model._model,self.spec)
        self.initial_gripper=env.robots[0].gripper['right'].current_action.copy()
        (self.path/'model.xml').write_text(env.sim.model.get_xml())
        initial_bytes=env.sim.get_state().flatten().tobytes()
        if env.initial_spec:initial_bytes+=json.dumps(env.initial_spec,sort_keys=True).encode()
        self.initial_state_hash=hashlib.sha256(initial_bytes).hexdigest()
        original=env._pre_action
        def hook(action,policy_step=False):
            original(action,policy_step=policy_step)
            if policy_step:self.captured_target=controller_target13(env).copy()
        env._pre_action=hook;self.original=original
        self.sample()
    def add(self,key,value):self.data.setdefault(key,[]).append(np.asarray(value).copy())
    def sample(self):
        e=self.env;d=e.sim.data;r=e.robots[0]
        synchronize_observation(e)
        rgb=e.sim.render(self.w,self.h,camera_name='preview')[::-1].copy();self.writer.append_data(rgb);self.add('observations/rgb',rgb)
        self.add('observations/timestamp',d.time);self.add('observations/qpos',d.qpos);self.add('observations/qvel',d.qvel)
        self.add('observations/robot_joint_position',d.qpos[r._ref_joint_pos_indexes]);self.add('observations/robot_joint_velocity',d.qvel[r._ref_joint_vel_indexes])
        self.add('observations/eef_executed13',executed13(e));self.add('observations/gripper_joint_position',d.qpos[r._ref_gripper_joint_pos_indexes['right']])
        self.add('observations/actuator_ctrl',d.ctrl)
        self.add('observations/object_pose_wxyz',np.r_[d.body_xpos[e.obj_body_id['cube']],d.body_xquat[e.obj_body_id['cube']]])
        self.add('observations/object_velocity',d.qvel[e.cube_dof:e.cube_dof+6])
        self.add('simulator/state',e.sim.get_state().flatten());integ=np.empty(self.nstate)
        mujoco.mj_getState(e.sim.model._model,d._data,integ,self.spec);self.add('simulator/integration_state',integ)
        labels,force,pairs=self.labeler.measure();self.add('events/flags',labels);self.add('events/gripper_cube_normal_force',force)
    def step(self,action,phase):
        self.times.append(float(self.env.sim.data.time));self.env.step(action)
        self.raw.append(np.asarray(action).copy());self.targets.append(self.captured_target.copy());self.phases.append(phase);self.sample()
    def finish(self,perturbation):
        self.writer.close();self.env._pre_action=self.original
        arrays={k:np.asarray(v) for k,v in self.data.items()};t=np.array(self.times)
        with h5py.File(self.path/'trajectory.h5','w') as f:
            for key,value in arrays.items():
                kwargs={'compression':'gzip','compression_opts':2} if value.ndim>1 else {}
                f.create_dataset(key,data=value,**kwargs)
            f['observations/timestep']=np.arange(len(arrays['observations/timestamp']))
            f['actions/timestep']=np.arange(len(self.raw))
            f['actions/raw']=np.asarray(self.raw);f['actions/target13']=np.asarray(self.targets);f['actions/timestamp']=t
            f.create_dataset('actions/phase',data=np.asarray(self.phases,dtype=h5py.string_dtype()))
            f['simulator/initial_gripper_action']=self.initial_gripper;f.attrs['mujoco_integration_state_spec']=self.spec
            f.attrs['observation_sampling']='mj_forward at current qpos; preserve full integration state; no time advance'
            f.attrs['event_names']=json.dumps(EVENT_NAMES);f.attrs['alignment']='state[i], rgb[i], action[i] -> state[i+1], rgb[i+1]'
        flags=arrays['events/flags'];timeline=[]
        for j,name in enumerate(EVENT_NAMES):
            starts=np.flatnonzero(flags[:,j] & ~np.r_[False,flags[:-1,j].astype(bool)])
            for idx in starts:timeline.append({'event':name,'state_index':int(idx),'time_seconds':float(arrays['observations/timestamp'][idx]-t[0])})
        success=bool(flags[-1,-1]);grasp=bool(flags[:,1].any());lift=bool(flags[:,2].any());drop=bool(flags[:,3].any())
        failure=None if success else ('dropped_before_goal' if drop else 'no_stable_grasp' if not grasp else 'no_lift_after_contact' if not lift else 'not_placed_at_goal')
        meta={'episode_id':self.path.name,'initial_state_id':'fixed_counter_0001','initial_state_sha256':self.initial_state_hash,'seed':20261003,'split':'train',
          'branch':self.kind,'success':success,'failure_type':failure,'actions':len(self.raw),'frames':len(flags),'duration_seconds':float(len(self.raw)/20),
          'perturbation':perturbation,'events':sorted(timeline,key=lambda x:x['state_index']),
          'event_summary':dict(zip(EVENT_NAMES,[bool(flags[:,j].any()) for j in range(len(EVENT_NAMES))])),
          'action_dimension':self.env.action_dim,'action_slices':dict(self.env.robots[0]._action_split_indexes),'goal_position_world':self.env.goal.tolist(),
          'task':'CounterCubePreview','standard_robocasa_benchmark':False,'robot':'PandaOmron (Franka Panda arm; base and torso commands zero)',
          'renderer':'MuJoCo EGL OpenGL','fps':20,'width':self.w,'height':self.h}
        # Perturbation times are observed execution phase boundaries.
        meta['phase_ranges']={p:{'first_action':self.phases.index(p),'last_action':len(self.phases)-1-self.phases[::-1].index(p)} for p in set(self.phases)}
        meta['attempt_results']=attempt_results(self.phases,flags,arrays['observations/timestamp'])
        for attempt in meta['attempt_results']:
            if not attempt['grasp_and_lift']:
                meta['events'].append({'event':'grasp_attempt_failed','state_index':attempt['end_state_index'],'time_seconds':attempt['time_seconds']})
        meta.update(self.episode_info)
        meta['actual_phase_intervals']=phase_intervals(self.phases,arrays['observations/timestamp'])
        if self.episode_info:
            pos=arrays['observations/object_pose_wxyz'];final=pos[-1,:3];contact=bool(flags[:,0].any())
            off=bool(final[2]<.80 or abs(final[0])>.78 or abs(final[1])>.35)
            meta['failure_type']=None if success else ('object_off_counter' if off else 'dropped_before_goal' if drop else 'grasp_miss' if not grasp and not contact else 'contact_without_lift' if not lift else 'lifted_but_not_placed')
            meta['outcome_metrics']={'max_object_height_m':float(pos[:,2].max()),'final_goal_xy_error_m':float(np.linalg.norm(final[:2]-self.env.goal[:2])), 'final_object_speed_m_s':float(np.linalg.norm(arrays['observations/object_velocity'][-1,:3])), 'object_off_counter':off}
        meta['events'].sort(key=lambda x:x['state_index'])
        (self.path/'metadata.json').write_text(json.dumps(meta,indent=2));return meta


def attempt_results(phases,flags,timestamps):
    indices=[i for i,p in enumerate(phases) if p=='lift' or p.endswith('/lift')];groups=[]
    for i in indices:
        if not groups or i!=groups[-1][-1]+1:groups.append([])
        groups[-1].append(i)
    return [{'attempt_index':j,'end_state_index':g[-1]+1,'time_seconds':float(timestamps[g[-1]+1]-timestamps[0]),
      'grasp_and_lift':bool(np.any(flags[np.array(g)+1,1].astype(bool)&flags[np.array(g)+1,2].astype(bool)))} for j,g in enumerate(groups)]


def phase_intervals(phases,timestamps):
    intervals=[];start=0
    for i in range(1,len(phases)+1):
        if i==len(phases) or phases[i]!=phases[start]:
            intervals.append({'phase':phases[start],'start_action_index':start,'end_action_index_exclusive':i,'start_time_seconds':float(timestamps[start]-timestamps[0]),'end_time_seconds':float(timestamps[i]-timestamps[0])});start=i
    return intervals
