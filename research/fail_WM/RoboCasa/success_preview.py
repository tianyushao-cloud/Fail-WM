"""Success-only multi-object previews. No perturbations or mid-rollout resets."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import h5py
import imageio.v2 as imageio
import mujoco
from PIL import Image, ImageDraw
from rc_collect.environment import CounterCubePreview
from rc_collect.action_conversion import native_action, controller_target13, executed13
from rc_collect.synchronization import synchronize_observation
from rc_collect.controller import DOWN
from robosuite.models.objects import BoxObject
from robosuite.controllers import load_composite_controller_config
import robosuite as suite

TASKS = {
    'clear_restore': {
        'instruction': 'Move the blue block to the blue staging area, place the red block on the green target, then return the blue block to the original yellow region on the counter.',
        'objects': {
            'cube': {'half': [.02]*3, 'color': [.85,.06,.04,1], 'position': [.59,-.19,.941]},
            'blue': {'half': [.022,.022,.02], 'color': [.04,.25,.85,1], 'position': [.59,-.19,.982]},
        },
        'markers': [([.47,-.20], [.05,.35,.95,.65]), ([.65,-.075], [.05,.75,.25,.65]), ([.59,-.19], [.95,.72,.06,.65])],
        'goals': {'cube': [.65,-.075,.940], 'blue': [.59,-.19,.940]},
        'moves': [('blue',[.47,-.20,.940]), ('cube',[.65,-.075,.940]), ('blue',[.59,-.19,.940])],
    },
    'stack': {
        'instruction': 'Place the blue base block on the green target, then stack the red block on the blue block and release both.',
        'objects': {
            'cube': {'half': [.02]*3, 'color': [.85,.06,.04,1], 'position': [.59,-.22,.941]},
            'blue': {'half': [.03,.03,.025], 'color': [.04,.25,.85,1], 'position': [.47,-.17,.946]},
        },
        'markers': [([.63,-.075], [.05,.75,.25,.65])],
        'goals': {'cube': [.63,-.075,.990], 'blue': [.63,-.075,.945]},
        'moves': [('blue',[.63,-.075,.945]), ('cube',[.63,-.075,.990])],
    },
    'sort_pair': {
        'instruction': 'Place one red block and one blue block in each outlined tray region. Keep all four blocks separate and release them.',
        'objects': {
            'cube': {'half': [.018]*3, 'color': [.85,.06,.04,1], 'position': [.46,-.25,.939]},
            'red2': {'half': [.018]*3, 'color': [.85,.06,.04,1], 'position': [.55,-.25,.939]},
            'blue': {'half': [.018]*3, 'color': [.04,.25,.85,1], 'position': [.64,-.25,.939]},
            'blue2': {'half': [.018]*3, 'color': [.04,.25,.85,1], 'position': [.46,-.14,.939]},
        },
        'markers': [],
        'goals': {'cube': [.54,-.075,.938], 'blue': [.54,-.14,.938], 'red2': [.66,-.075,.938], 'blue2': [.66,-.14,.938]},
        'moves': [('cube',[.54,-.075,.938]), ('blue',[.54,-.14,.938]), ('red2',[.66,-.075,.938]), ('blue2',[.66,-.14,.938])],
    },
}

class SuccessKitchenPreview(CounterCubePreview):
    def __init__(self, task='clear_restore', **kwargs):
        self.task_name = task
        self.definition = TASKS[task]
        super().__init__(initial_spec={'goal_position': next(iter(self.definition['goals'].values()))}, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        old = self.model.worldbody.find("geom[@name='place_target']")
        self.model.worldbody.remove(old)
        for i, (xy, color) in enumerate(self.definition['markers']):
            ET.SubElement(self.model.worldbody, 'geom', {'name': f'goal_{i}', 'type': 'cylinder', 'pos': f'{xy[0]} {xy[1]} .921', 'size': '.035 .0005', 'group': '1', 'rgba': ' '.join(map(str,color)), 'contype':'0','conaffinity':'0'})
        if self.task_name == 'sort_pair':
            for i,x in enumerate([.54,.66]):
                # Outlined target regions, not physical containers.
                for j,(pos,size) in enumerate([
                    ([x-.047,-.108,.9215],[.0015,.066,.0005]),
                    ([x+.047,-.108,.9215],[.0015,.066,.0005]),
                    ([x,-.174,.9215],[.047,.0015,.0005]),
                    ([x,-.042,.9215],[.047,.0015,.0005])]):
                    ET.SubElement(self.model.worldbody, 'geom', {'name':f'region_{i}_{j}','type':'box','pos':' '.join(map(str,pos)),'size':' '.join(map(str,size)),'group':'1','rgba':'.05 .65 .25 1','contype':'0','conaffinity':'0'})
        camera = self.model.worldbody.find("camera[@name='preview']")
        eye=np.array([.88,-.93,1.60]); target=np.array([.54,-.13,1.01])
        z=(eye-target)/np.linalg.norm(eye-target); x=np.cross([0,0,1],z); x/=np.linalg.norm(x); y=np.cross(z,x)
        camera.set('pos',' '.join(map(str,eye))); camera.set('xyaxes',' '.join(map(str,np.r_[x,y]))); camera.set('fovy','44')

    def _create_objects(self):
        self.objects={}
        self.object_cfgs=[]
        for name,spec in self.definition['objects'].items():
            obj=BoxObject(name,size=spec['half'],rgba=spec['color'],density=650,friction=[1.2,.005,.0001])
            self.objects[name]=obj
            self.object_cfgs.append({'name':name,'type':'object','init_robot_here':name=='cube','placement':{'fixture':self.counter,'size':(.5,.3),'pos':(0,0),'rotation':0},'info':{'cat':'cube','groups_containing_sampled_obj':['cube'],'mjcf_path':'procedural_box'}})
        self.model.merge_objects(list(self.objects.values()))

    def get_ep_meta(self):
        meta=super().get_ep_meta()
        meta['lang']=self.definition['instruction']; meta['task']=self.task_name
        return meta

    def pose(self,name):
        return self.sim.data.body_xpos[self.obj_body_id[name]].copy()

    def object_dof(self,name):
        j=self.sim.model.joint_name2id(self.objects[name].joints[0])
        return int(self.sim.model.jnt_dofadr[j])

    def predicates(self):
        if not hasattr(self,'ready'): return {'all_goals':False}
        d=self.sim.data; eef=d.site_xpos[self.robots[0].eef_site_id['right']]
        result={}
        for name,goal in self.definition['goals'].items():
            p=self.pose(name); dof=self.object_dof(name); vel=d.qvel[dof:dof+6]
            result[name+'_at_goal']=bool(np.linalg.norm(p[:2]-np.array(goal[:2]))<.018 and abs(p[2]-goal[2])<.009)
            result[name+'_stable']=bool(np.linalg.norm(vel[:3])<.03 and np.linalg.norm(vel[3:])<.2)
            result[name+'_released']=bool(np.linalg.norm(eef-p)>.10 and not self._check_grasp(self.robots[0].gripper['right'],self.objects[name]))
            support=self.objects['blue'] if self.task_name=='stack' and name=='cube' else self.counter
            result[name+'_supported']=bool(self.check_contact(self.objects[name],support))
        result['all_goals']=all(result.values())
        return result

    def _check_success(self):
        return self.predicates()['all_goals']


def make_env(task):
    cfg=load_composite_controller_config(robot='PandaOmron')
    cfg['body_parts']['right'].update({'type':'OSC_POSE','input_type':'delta','input_ref_frame':'world','kp':200,'damping_ratio':1,'impedance_mode':'fixed','output_max':[.04,.04,.04,.25,.25,.25],'output_min':[-.04,-.04,-.04,-.25,-.25,-.25]})
    e=suite.make('SuccessKitchenPreview',task=task,robots='PandaOmron',controller_configs=cfg,layout_ids=[1],style_ids=[1],generative_textures=False,randomize_cameras=False,use_distractors=False,clutter_mode=0,robot_spawn_deviation_pos_x=0,robot_spawn_deviation_pos_y=0,robot_spawn_deviation_rot=0,initialization_noise=None,has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,use_object_obs=True,camera_names=['preview'],camera_widths=640,camera_heights=480,render_gpu_device_id=0,control_freq=20,horizon=3000,ignore_done=True,seed=20261017)
    e.reset(); e.hard_reset=False
    from rc_collect.initial_state import prepare_initial_state
    prepare_initial_state(e)
    for name,spec in e.definition['objects'].items():
        e.sim.data.set_joint_qpos(e.objects[name].joints[0],np.r_[spec['position'],[1,0,0,0]])
    e.sim.data.qvel[:]=0; e.sim.forward(); e.ready=True
    r=e.robots[0]
    for c in r.part_controllers.values(): c.update(force=True); c.reset_goal()
    pos=e.sim.data.site_xpos[r.eef_site_id['right']].copy()
    rotation=e.sim.data.site_xmat[r.eef_site_id['right']].reshape(3,3).copy()
    for _ in range(30):
        e.step(native_action(e,pos,rotation,-1)); synchronize_observation(e)
    for c in r.part_controllers.values(): c.update(force=True); c.reset_goal()
    return e

class Recorder:
    def __init__(self,e,out):
        self.e=e; self.out=out; out.mkdir(parents=True,exist_ok=False)
        self.f=h5py.File(out/'trajectory.h5','w'); self.writer=imageio.get_writer(str(out/'rgb.mp4'),fps=20,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-threads','1'])
        self.names=list(e.objects); self.T=0; self.phases=[]; self.snapshots=[]; self.success_flags=[]; self.checkpoints=[]
        self.spec=int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.f.attrs['schema_version']='success_multiobject_v1'
        self.f.attrs['object_names']=json.dumps(self.names)
        self.f.attrs['task']=e.task_name
        self.f.attrs['mujoco_integration_state_spec']=self.spec
        self.f['simulator/initial_gripper_action']=e.robots[0].gripper['right'].current_action.copy()
        (out/'model.xml').write_text(e.sim.model.get_xml())
        self.original=e._pre_action
        def hook(action,policy_step=False):
            self.original(action,policy_step=policy_step)
            if policy_step:self.target=controller_target13(e).copy()
        e._pre_action=hook
        self.sample(); self.checkpoint('initial')

    def add(self,key,value):
        a=np.asarray(value)
        if key not in self.f:
            opts={'compression':'gzip','compression_opts':1} if a.ndim else {}
            self.f.create_dataset(key,shape=(0,)+a.shape,maxshape=(None,)+a.shape,chunks=(1,)+a.shape,dtype=a.dtype,**opts)
        ds=self.f[key]; ds.resize(ds.shape[0]+1,axis=0); ds[-1]=a

    def sample(self):
        e=self.e; synchronize_observation(e); d=e.sim.data
        rgb=e.sim.render(640,480,camera_name='preview')[::-1].copy()
        self.writer.append_data(rgb); self.add('observations/rgb',rgb)
        for name,value in [('qpos',d.qpos),('qvel',d.qvel),('timestamp',d.time),('eef_executed13',executed13(e)),('actuator_ctrl',d.ctrl)]:self.add('observations/'+name,value)
        self.add('observations/object_pose_wxyz',np.array([np.r_[e.pose(n),d.body_xquat[e.obj_body_id[n]]] for n in self.names]))
        self.add('observations/object_velocity',np.array([d.qvel[e.object_dof(n):e.object_dof(n)+6] for n in self.names]))
        state=np.empty(mujoco.mj_stateSize(e.sim.model._model,self.spec)); mujoco.mj_getState(e.sim.model._model,d._data,state,self.spec)
        self.add('simulator/integration_state',state)
        pred=e.predicates();self.add('events/goal_state',np.array(list(pred.values()),dtype=np.uint8));self.f.attrs['predicate_names']=json.dumps(list(pred))
        self.success_flags.append(pred['all_goals'])

    def step(self,action,phase):
        self.add('actions/timestamp',self.e.sim.data.time)
        self.e.step(action)
        self.add('actions/raw',action); self.add('actions/target13',self.target)
        self.phases.append(phase); self.T+=1; self.sample()
        if self.T>2400: raise RuntimeError('Explicit 120 second rollout limit reached')

    def checkpoint(self,label):
        self.snapshots.append((label,self.f['observations/rgb'][-1].copy()))
        self.checkpoints.append({'label':label,'state_index':self.T,'time_seconds':self.T/20,'poses':{n:self.e.pose(n).tolist() for n in self.names}})

    def finish(self,controller):
        self.e._pre_action=self.original;self.writer.close()
        self.f.create_dataset('actions/phase',data=np.asarray(self.phases,dtype=h5py.string_dtype()))
        success=bool(len(self.success_flags)>=21 and all(self.success_flags[-21:]) and all(x['passed'] for x in controller.subgoals))
        meta={'task':self.e.task_name,'instruction':self.e.definition['instruction'],'success':success,'stable_final_window_seconds':1.0,'actions':self.T,'frames':self.T+1,'duration_seconds':self.T/20,'object_names':self.names,'task_definition':self.e.definition,'subgoals':controller.subgoals,'phase_timeouts':controller.timeouts,'checkpoints':self.checkpoints,'final_predicates':self.e.predicates(),'perturbations':[],'recovery_enabled':False,'reset_during_rollout':False,'privileged_scripted_controller':True,'standard_robocasa_benchmark':False,'seed':20261017,'split':'preview_only'}
        self.f.close();(self.out/'metadata.json').write_text(json.dumps(meta,indent=2))
        images=self.snapshots
        sheet=Image.new('RGB',(640*min(3,len(images)),510*((len(images)+2)//3)),(245,245,245));draw=ImageDraw.Draw(sheet)
        for i,(label,img) in enumerate(images):
            x=(i%3)*640;y=(i//3)*510;sheet.paste(Image.fromarray(img),(x,y));draw.text((x+12,y+485),label,fill=(20,20,20))
        sheet.save(self.out/'contact_sheet.jpg',quality=92)
        return meta

class Controller:
    def __init__(self,e,rec):self.e=e;self.rec=rec;self.subgoals=[];self.timeouts=[]
    def eef(self):return self.e.sim.data.site_xpos[self.e.robots[0].eef_site_id['right']].copy()
    def move(self,p,g,phase,max_steps=110):
        p=np.array(p); stable=0
        for i in range(max_steps):
            self.rec.step(native_action(self.e,p,DOWN,g),phase)
            stable=stable+1 if np.linalg.norm(self.eef()-p)<.006 else 0
            if i>=12 and stable>=6:break
        if stable<6:self.timeouts.append({'phase':phase,'error':float(np.linalg.norm(self.eef()-p))})
        print(phase,'steps',i+1,'error',round(float(np.linalg.norm(self.eef()-p)),5),flush=True)
    def hold(self,p,g,steps,phase):
        for _ in range(steps):self.rec.step(native_action(self.e,p,DOWN,g),phase)
    def transfer(self,name,goal,index):
        prefix=f'{index}_{name}';p=self.e.pose(name)
        self.move(np.r_[p[:2],1.18],-1,prefix+'/approach')
        self.move(p,-1,prefix+'/descend');self.hold(p,1,22,prefix+'/grasp')
        self.move(np.r_[p[:2],1.18],1,prefix+'/lift')
        lifted=bool(self.e.pose(name)[2]>p[2]+.08 and self.e._check_grasp(self.e.robots[0].gripper['right'],self.e.objects[name]))
        if not lifted:raise RuntimeError(f'Failed to lift {name}; stop this calibration rollout')
        actual_goal=np.array(goal,float)
        if self.e.task_name=='stack' and name=='cube':actual_goal=self.e.pose('blue')+np.array([0,0,.045])
        # Use the measured gripper/object offset so destination is object-centered.
        offset=self.eef()-self.e.pose(name)
        self.move(np.r_[actual_goal[:2]+offset[:2],1.18],1,prefix+'/transport')
        target=actual_goal+offset+[0,0,.004]
        self.move(target,1,prefix+'/lower');self.hold(target,-1,20,prefix+'/release')
        self.move(np.r_[target[:2],1.18],-1,prefix+'/retreat');self.hold(self.eef(),-1,25,prefix+'/settle')
        pos=self.e.pose(name);passed=bool(np.linalg.norm(pos[:2]-actual_goal[:2])<.018 and abs(pos[2]-actual_goal[2])<.009)
        if self.e.task_name=='stack' and name=='cube':passed=bool(passed and self.e.check_contact(self.e.objects['cube'],self.e.objects['blue']))
        self.subgoals.append({'step':index,'object':name,'goal':actual_goal.tolist(),'actual_position':pos.tolist(),'lifted':lifted,'passed':passed})
        self.rec.checkpoint(prefix+'_placed')
        print('SUBGOAL',self.subgoals[-1],flush=True)
        if not passed:raise RuntimeError(f'Placement subgoal failed: {name}')
    def run(self):
        for i,(name,goal) in enumerate(self.e.definition['moves'],1):self.transfer(name,goal,i)
        self.move([.42,-.28,1.20],-1,'final_clear_view');self.hold(self.eef(),-1,45,'final_stability')
        self.rec.checkpoint('final_stable')


def validate(path,task):
    e=make_env(task)
    try:
        with h5py.File(path/'trajectory.h5','r') as f:
            actions=f['actions/raw'][:];T=len(actions);ts=f['observations/timestamp'][:]
            assert actions.shape==(T,12) and f['actions/target13'].shape==(T,13)
            assert all(len(d)==T+1 for d in f['observations'].values())
            assert all(len(d)==T for d in f['actions'].values())
            assert np.allclose(np.diff(ts),.05,atol=1e-9,rtol=0)
            assert np.allclose(ts[:-1],f['actions/timestamp'][:],atol=1e-9,rtol=0)
            assert np.isfinite(actions).all()
            r=f['actions/target13'][:,3:12].reshape(-1,3,3)
            assert np.max(abs(r@r.transpose(0,2,1)-np.eye(3)))<1e-6
            assert np.max(abs(np.linalg.det(r)-1))<1e-6
            spec=int(f.attrs['mujoco_integration_state_spec']);initial=f['simulator/integration_state'][0]
            mujoco.mj_setState(e.sim.model._model,e.sim.data._data,initial,spec);e.sim.forward()
            e.robots[0].gripper['right'].current_action=f['simulator/initial_gripper_action'][:]
            for c in e.robots[0].part_controllers.values():c.update(force=True);c.reset_goal()
            mujoco.mj_setState(e.sim.model._model,e.sim.data._data,initial,spec)
            original=e._pre_action;captured=[]
            def hook(action,policy_step=False):
                original(action,policy_step=policy_step)
                if policy_step:captured.append(controller_target13(e).copy())
            e._pre_action=hook
            errors={'qpos':0.,'qvel':0.,'object_pose':0.,'executed13':0.,'rgb_mae':0.}
            names=json.loads(f.attrs['object_names']);samples=set(np.linspace(0,T,7,dtype=int));goal_flags=[]
            for i in range(T+1):
                if i:e.step(actions[i-1])
                synchronize_observation(e)
                comparisons={'qpos':(e.sim.data.qpos,f['observations/qpos'][i]),'qvel':(e.sim.data.qvel,f['observations/qvel'][i]),'object_pose':(np.array([np.r_[e.pose(n),e.sim.data.body_xquat[e.obj_body_id[n]]] for n in names]),f['observations/object_pose_wxyz'][i]),'executed13':(executed13(e),f['observations/eef_executed13'][i])}
                for k,(a,b) in comparisons.items():errors[k]=max(errors[k],float(np.max(abs(a-b))))
                predicates=e.predicates();goal_flags.append(predicates['all_goals'])
                assert np.array_equal(np.array(list(predicates.values()),dtype=np.uint8),f['events/goal_state'][i])
                if i in samples:
                    rgb=e.sim.render(640,480,camera_name='preview')[::-1]
                    errors['rgb_mae']=max(errors['rgb_mae'],float(np.mean(abs(rgb.astype(float)-f['observations/rgb'][i]))))
            errors['target13']=float(np.max(abs(np.array(captured)-f['actions/target13'][:])))
            assert errors['qpos']<1e-6 and errors['qvel']<1e-5 and errors['object_pose']<1e-6 and errors['executed13']<1e-6 and errors['target13']<1e-6 and errors['rgb_mae']<.05,errors
            assert all(goal_flags[-21:]),'Final stable success missing on replay'
        reader=imageio.get_reader(str(path/'rgb.mp4'));vm=reader.get_meta_data();frames=sum(1 for _ in reader);reader.close()
        assert frames==T+1 and abs(vm['fps']-20)<1e-6
        report={'passed':True,'alignment_passed':True,'replay_passed':True,'success_stable_on_replay':True,'video_frames':frames,'errors':errors}
        (path/'validation.json').write_text(json.dumps(report,indent=2));return report
    finally:e.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=list(TASKS),default='clear_restore');ap.add_argument('--output',required=True);ap.add_argument('--validate-only',action='store_true');a=ap.parse_args();out=Path(a.output)
    if a.validate_only:print(json.dumps(validate(out,a.task)));return
    e=make_env(a.task);rec=None
    try:
        rec=Recorder(e,out);controller=Controller(e,rec);controller.run();meta=rec.finish(controller);rec=None
        print('RESULT',json.dumps(meta),flush=True)
        if not meta['success']:raise RuntimeError('Final success predicates were not stable')
    finally:
        if rec is not None:
            e._pre_action=rec.original;rec.writer.close();rec.f.close()
        e.close()
    print('VALIDATION',json.dumps(validate(out,a.task)),flush=True)

if __name__=='__main__':main()
