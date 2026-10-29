"""Non-block RoboCasa Kitchen previews with physical drawer and push/grasp tasks.

Procedural fixtures and privileged state-feedback control. No attachments,
kinematic object animation, mid-rollout resets, or outcome relabeling.
"""
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
import robosuite as suite
from robosuite.models.objects import CanObject, CylinderObject
from robosuite.controllers import load_composite_controller_config
from rc_collect.environment import CounterCubePreview
from rc_collect.action_conversion import native_action, controller_target13, executed13
from rc_collect.synchronization import synchronize_observation
from rc_collect.controller import DOWN
from success_preview import Recorder as BaseRecorder, Controller as BaseController

SEED = 20261022
TASKS = {
    'drawer_can': {
        'instruction': 'Grasp the drawer handle, pull the drawer open, take the soda can out and place it upright on the green target, then close the drawer.',
        'goals': {'can': [.44, -.14, .9603]},
        'initial': [.60, -.02, 1.01],
        'fixture_type': 'procedural sliding drawer with physical handle and fixed roof',
    },
    'push_then_bin': {
        'instruction': 'Push the round orange puck along the guide lane to the open pickup area, then grasp it and place it inside the green physical bin.',
        'goals': {'puck': [.66, -.075, .943]},
        'initial': [.47, -.23, .933],
        'fixture_type': 'physical guide rails and five-geom open bin',
    },
}

def numbers(values):
    return ' '.join(map(str, values))

def geom(parent, name, pos, size, color, kind='box', **kwargs):
    attrs = dict(name=name, type=kind, pos=numbers(pos), size=numbers(size),
                 rgba=numbers(color), group='1', friction='1 .005 .0001',
                 solref='.01 1', condim='4')
    attrs.update({k: str(v) for k,v in kwargs.items()})
    return ET.SubElement(parent, 'geom', attrs)

class ComplexInteractionKitchen(CounterCubePreview):
    def __init__(self, task='drawer_can', **kwargs):
        self.task_name=task
        self.definition=TASKS[task]
        self.ready=False
        super().__init__(initial_spec={'goal_position': next(iter(self.definition['goals'].values()))}, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        w=self.model.worldbody
        w.remove(w.find("geom[@name='place_target']"))
        if self.task_name=='drawer_can':
            x,y=.60,-.02
            # Fixed housing. The roof prevents direct top-down access while closed.
            geom(w,'housing_left',[x-.078,y,.995],[.008,.088,.075],[.30,.34,.40,1])
            geom(w,'housing_right',[x+.078,y,.995],[.008,.088,.075],[.30,.34,.40,1])
            geom(w,'housing_back',[x,y+.084,.995],[.07,.008,.075],[.30,.34,.40,1])
            geom(w,'housing_roof',[x,y,1.105],[.086,.092,.008],[.42,.46,.52,1])
            d=ET.SubElement(w,'body',name='drawer_body',pos=numbers([x,y,.945]))
            ET.SubElement(d,'inertial',pos='0 0 .025',mass='.30',diaginertia='.001 .001 .001')
            ET.SubElement(d,'joint',name='drawer_slide',type='slide',axis='0 -1 0',range='0 .15',limited='true',damping='3',frictionloss='.3',armature='.01')
            geom(d,'drawer_floor',[0,0,0],[.065,.075,.005],[.68,.49,.28,1],density=350)
            geom(d,'drawer_left',[-.063,0,.035],[.004,.075,.035],[.68,.49,.28,1],density=350)
            geom(d,'drawer_right',[.063,0,.035],[.004,.075,.035],[.68,.49,.28,1],density=350)
            geom(d,'drawer_back',[0,.072,.035],[.065,.004,.035],[.68,.49,.28,1],density=350)
            geom(d,'drawer_front',[0,-.08,.045],[.064,.006,.045],[.75,.57,.34,1],density=350)
            geom(d,'handle_stem',[0,-.130,.055],[.008,.050,.008],[.12,.15,.18,1],density=500)
            geom(d,'drawer_handle',[0,-.178,.055],[.012,.026],[.12,.15,.18,1],kind='cylinder',density=500)
            ET.SubElement(d,'site',name='handle_grasp',pos='0 -.178 .055',size='.004',rgba='0 0 0 0')
            geom(w,'can_target',[.44,-.14,.921],[.05,.001],[.05,.7,.2,.7],kind='cylinder',contype=0,conaffinity=0)
        else:
            for i,yy in enumerate([-.276,-.184]):
                geom(w,f'guide_{i}',[.53,yy,.937],[.085,.004,.017],[.18,.35,.65,1])
            # Bin is collision geometry, not a painted region.
            x,y=.66,-.075
            geom(w,'bin_floor',[x,y,.925],[.06,.06,.006],[.08,.55,.25,1])
            for i,(p,s) in enumerate([
                ([x-.064,y,.948],[.004,.064,.023]),([x+.064,y,.948],[.004,.064,.023]),
                ([x,y-.064,.948],[.06,.004,.023]),([x,y+.064,.948],[.06,.004,.023])]):
                geom(w,f'bin_wall_{i}',p,s,[.08,.65,.30,1])
            geom(w,'pickup_mark',[.625,-.23,.921],[.027,.001],[.9,.8,.1,.6],kind='cylinder',contype=0,conaffinity=0)
        camera=w.find("camera[@name='preview']")
        eye=np.array([.95,-1.02,1.68]); target=np.array([.555,-.11,1.015])
        z=(eye-target)/np.linalg.norm(eye-target);xx=np.cross([0,0,1],z);xx/=np.linalg.norm(xx);yy=np.cross(z,xx)
        camera.set('pos',numbers(eye));camera.set('xyaxes',numbers(np.r_[xx,yy]));camera.set('fovy','47')

    def _create_objects(self):
        if self.task_name=='drawer_can':
            name='can';obj=CanObject(name)
            obj.size=np.array([.05,.05,.10])
        else:
            name='puck';obj=CylinderObject(name,size=[.025,.012],rgba=[.95,.30,.035,1],density=650,friction=[.45,.005,.0001])
        self.objects={name:obj}
        self.object_cfgs=[{'name':name,'type':'object','init_robot_here':True,
            'placement':{'fixture':self.counter,'size':(.5,.3),'pos':(0,0),'rotation':0},
            'info':{'cat':name,'groups_containing_sampled_obj':[name],'mjcf_path':'robosuite_can' if name=='can' else 'procedural_cylinder'}}]
        self.model.merge_objects([obj])

    def get_ep_meta(self):
        return {'lang':self.definition['instruction'],'task':self.task_name,'custom_task':True,
                'layout_variant':'minimal procedural RoboCasa counter'}

    def edit_model_xml(self,xml_str):
        # Kitchen rewrites asset paths into its own tree. Retain existing
        # robosuite can meshes/textures when RoboCasa has no counterpart.
        root=ET.fromstring(super().edit_model_xml(xml_str))
        suite_assets=Path(suite.__file__).parent/'models/assets'
        for el in root.find('asset'):
            value=el.get('file')
            if value and not Path(value).is_file() and '/models/assets/' in value:
                candidate=suite_assets/value.split('/models/assets/',1)[1]
                if candidate.is_file():el.set('file',str(candidate))
        return ET.tostring(root,encoding='unicode')

    def pose(self,name):
        return self.sim.data.body_xpos[self.obj_body_id[name]].copy()

    def object_dof(self,name):
        j=self.sim.model.joint_name2id(self.objects[name].joints[0])
        return int(self.sim.model.jnt_dofadr[j])

    def drawer_position(self):
        return float(self.sim.data.get_joint_qpos('drawer_slide'))

    def fixture_state(self):
        if self.task_name=='drawer_can':
            j=self.sim.model.joint_name2id('drawer_slide')
            return np.array([self.drawer_position(),self.sim.data.qvel[self.sim.model.jnt_dofadr[j]]])
        return np.empty(0)

    def contact_flags(self):
        r=self.robots[0];gripper=r.gripper['right'].contact_geoms
        name=next(iter(self.objects));obj=self.objects[name]
        target=['drawer_handle','handle_stem'] if self.task_name=='drawer_can' else ['guide_0','guide_1']
        return np.array([self.check_contact(gripper,obj),self.check_contact(gripper,target)],dtype=np.uint8)

    def predicates(self):
        if not self.ready:return {'all_goals':False}
        name=next(iter(self.objects));p=self.pose(name);goal=np.array(self.definition['goals'][name])
        dof=self.object_dof(name);v=self.sim.data.qvel[dof:dof+6]
        eef=self.sim.data.site_xpos[self.robots[0].eef_site_id['right']]
        result={
            'object_at_goal':bool(np.linalg.norm(p[:2]-goal[:2])<.025 and abs(p[2]-goal[2])<.012),
            'object_stable':bool(np.linalg.norm(v[:3])<.03 and np.linalg.norm(v[3:])<.2),
            'object_released':bool(np.linalg.norm(eef-p)>.10 and not self._check_grasp(self.robots[0].gripper['right'],self.objects[name])),
            'object_upright':bool(self.sim.data.body_xmat[self.obj_body_id[name]].reshape(3,3)[2,2]>.96),
        }
        if self.task_name=='drawer_can':
            result['object_supported']=bool(self.check_contact(self.objects[name],self.counter))
            result['drawer_closed']=self.drawer_position()<.008
        else:
            result['object_supported']=bool(self.check_contact(self.objects[name],['bin_floor']))
            result['object_inside_bin']=bool(abs(p[0]-.66)+.025<.060 and abs(p[1]+.075)+.025<.060)
        result['all_goals']=all(result.values())
        return result

    def _check_success(self):
        return self.predicates()['all_goals']

def make_env(task):
    cfg=load_composite_controller_config(robot='PandaOmron')
    cfg['body_parts']['right'].update({'type':'OSC_POSE','input_type':'delta','input_ref_frame':'world','kp':200,'damping_ratio':1,'impedance_mode':'fixed','output_max':[.04,.04,.04,.25,.25,.25],'output_min':[-.04,-.04,-.04,-.25,-.25,-.25]})
    e=suite.make('ComplexInteractionKitchen',task=task,robots='PandaOmron',controller_configs=cfg,
        layout_ids=[1],style_ids=[1],generative_textures=False,randomize_cameras=False,use_distractors=False,clutter_mode=0,
        robot_spawn_deviation_pos_x=0,robot_spawn_deviation_pos_y=0,robot_spawn_deviation_rot=0,initialization_noise=None,
        has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,use_object_obs=True,camera_names=['preview'],
        camera_widths=640,camera_heights=480,render_gpu_device_id=0,control_freq=20,horizon=4000,ignore_done=True,seed=SEED)
    e.reset();e.hard_reset=False
    from robocasa.utils.env_utils import set_robot_to_position
    set_robot_to_position(e,np.array([.30,-.60,0.]));e.init_robot_base_pos=np.array([.30,-.60,0.])
    r=e.robots[0]
    e.sim.data.qpos[r._ref_joint_pos_indexes]=r.init_qpos
    e.sim.data.qpos[r._ref_gripper_joint_pos_indexes['right']]=[.04,-.04]
    for name,obj in e.objects.items():e.sim.data.set_joint_qpos(obj.joints[0],np.r_[e.definition['initial'],[1,0,0,0]])
    if task=='drawer_can':e.sim.data.set_joint_qpos('drawer_slide',0.)
    e.sim.data.qvel[:]=0;e.sim.forward();r.gripper['right'].current_action=np.array([1.,-1.])
    for c in r.part_controllers.values():c.update(force=True);c.reset_goal()
    pos=e.sim.data.site_xpos[r.eef_site_id['right']].copy()
    ori=e.sim.data.site_xmat[r.eef_site_id['right']].reshape(3,3).copy()
    for _ in range(30):e.step(native_action(e,pos,ori,-1));synchronize_observation(e)
    for c in r.part_controllers.values():c.update(force=True);c.reset_goal()
    e.ready=True
    return e

class Recorder(BaseRecorder):
    def sample(self):
        super().sample()
        if self.e.task_name=='drawer_can':self.add('observations/fixture_position_velocity',self.e.fixture_state())
        self.add('events/gripper_contacts',self.e.contact_flags())

    def finish(self,controller):
        meta=super().finish(controller)
        meta.update({'seed':SEED,'schema_version':'complex_interaction_v1',
            'fixture_type':self.e.definition['fixture_type'],
            'fixture_joint_names':['drawer_slide'] if self.e.task_name=='drawer_can' else [],
            'contact_flag_names':['gripper_object','gripper_handle_or_guides'],
            'source_script':'complex_interactions_20261022.py'})
        (self.out/'metadata.json').write_text(json.dumps(meta,indent=2))
        return meta

class Controller(BaseController):
    def goal(self,label,passed,**values):
        row={'label':label,'passed':bool(passed),**values}
        self.subgoals.append(row);print('SUBGOAL',json.dumps(row),flush=True)
        if not passed:raise RuntimeError('Physical subgoal failed: '+label)
        self.rec.checkpoint(label)

    def transfer_object(self,name,goal):
        p=self.e.pose(name)
        self.move(np.r_[p[:2],1.24],-1,'object/approach')
        self.move(p,-1,'object/descend');self.hold(p,1,24,'object/grasp')
        self.move(np.r_[p[:2],1.24],1,'object/lift')
        grasp=self.e._check_grasp(self.e.robots[0].gripper['right'],self.e.objects[name])
        self.goal('object_lifted',self.e.pose(name)[2]>p[2]+.08 and grasp,position=self.e.pose(name).tolist())
        offset=self.eef()-self.e.pose(name);goal=np.array(goal)
        self.move(np.r_[goal[:2]+offset[:2],1.24],1,'object/transport')
        target=goal+offset+[0,0,.003]
        self.move(target,1,'object/lower');self.hold(target,-1,22,'object/release')
        self.move(np.r_[target[:2],1.24],-1,'object/retreat');self.hold(self.eef(),-1,25,'object/settle')
        p=self.e.pose(name)
        self.goal('object_placed',np.linalg.norm(p[:2]-goal[:2])<.025 and abs(p[2]-goal[2])<.012,position=p.tolist())

    def drawer(self,opening):
        phase='open' if opening else 'close'
        sid=self.e.sim.model.site_name2id('handle_grasp');h=self.e.sim.data.site_xpos[sid].copy()
        self.move(np.r_[h[:2],1.24],-1,f'drawer_{phase}/approach')
        self.move(h,-1,f'drawer_{phase}/descend');self.hold(h,1,22,f'drawer_{phase}/grasp_handle')
        contact=bool(self.e.check_contact(self.e.robots[0].gripper['right'].contact_geoms,['drawer_handle']))
        q0=self.e.drawer_position();target_q=.148 if opening else .001
        target=h+np.array([0,-(target_q-q0),0])
        # Slow continuous pose control. No joint actuator or direct state edits.
        for i in range(90):
            p=h+(target-h)*min(1.,(i+1)/70)
            self.rec.step(native_action(self.e,p,DOWN,1),f'drawer_{phase}/slide')
        self.hold(target,1,15,f'drawer_{phase}/settle')
        q=self.e.drawer_position()
        self.goal('drawer_'+('opened' if opening else 'closed'),contact and (q>.13 if opening else q<.008),slide_position=q,handle_contact=contact)
        actual_h=self.e.sim.data.site_xpos[sid].copy()
        self.hold(actual_h,-1,15,f'drawer_{phase}/release')
        self.move(np.r_[actual_h[:2],1.24],-1,f'drawer_{phase}/retreat')

    def push(self):
        p=self.e.pose('puck').copy()
        self.move([p[0]-.065,p[1],1.16],1,'push/approach')
        start=np.array([p[0]-.050,p[1],.942])
        self.move(start,1,'push/descend_behind')
        contact=False
        for i in range(100):
            target=start+np.array([.175*min(1.,(i+1)/90),0,0])
            self.rec.step(native_action(self.e,target,DOWN,1),'push/through_lane')
            contact=contact or bool(self.e.check_contact(self.e.robots[0].gripper['right'].contact_geoms,self.e.objects['puck']))
        after=self.e.pose('puck').copy()
        self.goal('puck_pushed_through_lane',contact and after[0]>.615 and abs(after[1]+.23)<.025 and abs(after[2]-.933)<.012,
                  before=p.tolist(),after=after.tolist(),gripper_contact=contact)
        self.move([self.eef()[0],self.eef()[1],1.19],1,'push/retreat')
        self.hold(self.eef(),-1,15,'push/open_for_pickup')

    def run(self):
        if self.e.task_name=='drawer_can':
            self.drawer(True)
            self.transfer_object('can',self.e.definition['goals']['can'])
            self.drawer(False)
        else:
            self.push();self.transfer_object('puck',self.e.definition['goals']['puck'])
        self.move([.40,-.31,1.25],-1,'final/clear_view')
        self.hold(self.eef(),-1,45,'final/stability');self.rec.checkpoint('final_stable')

def validate(path,task):
    e=make_env(task)
    try:
        with h5py.File(path/'trajectory.h5','r') as f:
            actions=f['actions/raw'][:];T=len(actions);ts=f['observations/timestamp'][:]
            assert actions.shape==(T,12) and f['actions/target13'].shape==(T,13)
            assert all(len(d)==T+1 for d in f['observations'].values())
            assert all(len(d)==T for d in f['actions'].values())
            assert len(f['events/goal_state'])==T+1 and len(f['events/gripper_contacts'])==T+1
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
            errors={k:0. for k in ['qpos','qvel','object_pose','executed13','rgb_mae','fixture']}
            names=json.loads(f.attrs['object_names']);samples=set(np.linspace(0,T,7,dtype=int));goal_flags=[]
            for i in range(T+1):
                if i:e.step(actions[i-1])
                synchronize_observation(e)
                comparisons={'qpos':(e.sim.data.qpos,f['observations/qpos'][i]),'qvel':(e.sim.data.qvel,f['observations/qvel'][i]),
                    'object_pose':(np.array([np.r_[e.pose(n),e.sim.data.body_xquat[e.obj_body_id[n]]] for n in names]),f['observations/object_pose_wxyz'][i]),
                    'executed13':(executed13(e),f['observations/eef_executed13'][i])}
                if task=='drawer_can':comparisons['fixture']=(e.fixture_state(),f['observations/fixture_position_velocity'][i])
                for k,(a,b) in comparisons.items():errors[k]=max(errors[k],float(np.max(abs(a-b))))
                pred=e.predicates();goal_flags.append(pred['all_goals'])
                assert np.array_equal(np.array(list(pred.values()),dtype=np.uint8),f['events/goal_state'][i])
                assert np.array_equal(e.contact_flags(),f['events/gripper_contacts'][i])
                if i in samples:
                    rgb=e.sim.render(640,480,camera_name='preview')[::-1]
                    errors['rgb_mae']=max(errors['rgb_mae'],float(np.mean(abs(rgb.astype(float)-f['observations/rgb'][i]))))
            errors['target13']=float(np.max(abs(np.array(captured)-f['actions/target13'][:])))
            assert all(errors[k]<(1e-5 if k=='qvel' else .05 if k=='rgb_mae' else 1e-6) for k in errors),errors
            meta=json.loads((path/'metadata.json').read_text())
            assert all(goal_flags[-21:]) and all(x['passed'] for x in meta['subgoals'])
            # Recompute ordered intermediate physical milestones from replay-matched states.
            p=f['observations/object_pose_wxyz'][:,0,:3];contacts=f['events/gripper_contacts'][:]
            if task=='drawer_can':
                q=f['observations/fixture_position_velocity'][:,0]
                opened=np.flatnonzero(q>.13);lifted=np.flatnonzero(p[:,2]>1.09)
                assert len(opened) and len(lifted) and opened[0]<lifted[0] and q[-1]<.008
                assert np.any(contacts[:,1]),'No physical handle contact'
            else:
                pushed=np.flatnonzero((p[:,0]>.615)&(abs(p[:,2]-.933)<.012))
                lifted=np.flatnonzero(p[:,2]>1.06)
                assert len(pushed) and len(lifted) and pushed[0]<lifted[0]
                assert np.any(contacts[:pushed[0]+1,0]),'No physical pushing contact'
        reader=imageio.get_reader(str(path/'rgb.mp4'));vm=reader.get_meta_data();frames=sum(1 for _ in reader);reader.close()
        assert frames==T+1 and abs(vm['fps']-20)<1e-6
        report={'passed':True,'alignment_passed':True,'replay_passed':True,'ordered_physical_milestones_passed':True,
            'success_stable_on_replay':True,'video_frames':frames,'errors':errors}
        (path/'validation.json').write_text(json.dumps(report,indent=2));return report
    finally:e.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=list(TASKS),required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--validate-only',action='store_true');a=ap.parse_args();out=Path(a.output)
    if a.validate_only:print(json.dumps(validate(out,a.task)));return
    e=make_env(a.task);rec=None
    try:
        rec=Recorder(e,out);c=Controller(e,rec);c.run();meta=rec.finish(c);rec=None
        print('RESULT',json.dumps(meta),flush=True)
        if not meta['success']:raise RuntimeError('Final success predicates were not stable')
    finally:
        if rec is not None:e._pre_action=rec.original;rec.writer.close();rec.f.close()
        e.close()
    print('VALIDATION',json.dumps(validate(out,a.task)),flush=True)

if __name__=='__main__':main()
