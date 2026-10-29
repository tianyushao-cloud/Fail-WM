"""A minimal custom task on RoboCasa Kitchen + its procedural kitchen fixtures.

This is deliberately not presented as a standard RoboCasa365 benchmark task.
MuJoCo simulates all contacts; no object attachment or trajectory teleportation.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np
import robosuite as suite
import robocasa
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robocasa.models.fixtures import Counter
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import BoxObject
from robosuite.controllers import load_composite_controller_config
from robocasa.utils import camera_utils as CamUtils

class CounterCubePreview(Kitchen):
    def __init__(self, initial_spec=None, **kwargs):
        self.initial_spec=dict(initial_spec or {})
        super().__init__(**kwargs)
    def _setup_model(self):
        # Native RoboCasa scene/fixture builder, fixed minimal layout.
        self.layout_id=1; self.style_id=1; self._curr_gen_fixtures=None
        for robot in self.robots:
            robot.init_qpos=np.array([-.01613,-1.03447,-.02398,-2.27551,.03932,1.51639,.69616])
            if hasattr(robot,'init_torso_qpos'):robot.init_torso_qpos=np.array([.15])
        ar=Path(robocasa.models.assets_root)
        texture_dir=Path(__file__).resolve().parents[1]/'assets/textures'
        texture_dir.mkdir(parents=True,exist_ok=True)
        import shutil
        texture=texture_dir/'marble.png'
        if not texture.exists():shutil.copyfile(ar/'fixtures/counters/counter/visuals/marble.png',texture)
        marble=str(texture)
        wood=str(Path(suite.__file__).parent/'models/assets/textures/wood-light.png')
        # Use bundled textures to keep the five-episode prototype self-contained.
        if not Path(wood).exists():wood=marble
        layout={'main_group':{'counter':[{'name':'counter','type':'counter','size':[1.5,.65,.92], 'pos':[0,0,.46], 'top_texture':marble, 'base_texture':wood,'base_color':[.6,.5,.4,1],'island_only':True}],
                              'wall':[{'name':'backwall','type':'wall','size':[1.3,.85,.025],'pos':[0,.34,.85],'texture':marble}],
                              'floor':[{'name':'floor','type':'floor','size':[2,2,.02],'pos':[0,0,0],'texture':wood}]}}
        self.mujoco_arena=KitchenArena(deepcopy(layout),{},rng=self.rng)
        self.mujoco_arena.set_origin([0,0,0]);CamUtils.set_cameras(self)
        self.fixture_cfgs=self.mujoco_arena.get_fixture_cfgs();self.fixtures={c['name']:c['model'] for c in self.fixture_cfgs}
        self.model=ManipulationTask(self.mujoco_arena,[r.robot_model for r in self.robots],list(self.fixtures.values()),enable_multiccd=True,enable_sleeping_islands=False)
        # Fixed destination disc is a real scene visual, not a video overlay.
        for light in self.model.worldbody.iter('light'):
            light.set('ambient','.08 .08 .08');light.set('diffuse','.35 .35 .35');light.set('specular','.04 .04 .04')
        for mat in self.model.asset.iter('material'):
            mat.set('reflectance','0');mat.set('shininess','.05');mat.set('specular','.1')
        self.goal=np.array(self.initial_spec.get('goal_position',[.66,-.10,.942]),dtype=float)
        ET.SubElement(self.model.worldbody,'geom',{'name':'place_target','type':'cylinder','pos':f'{self.goal[0]} {self.goal[1]} .921','size':'.055 .001','group':'1','rgba':'.05 .65 .25 .8','contype':'0','conaffinity':'0'})
        ET.SubElement(self.model.worldbody,'camera',{'name':'preview','mode':'fixed','pos':'.9 -1.4 1.9','xyaxes':'.83205 .5547 0 -.2691 .40365 .8745','fovy':'48'})
        visual=self.model.root.find('visual')
        if visual is not None and visual.find('map') is not None:visual.find('map').set('zfar','15')
        ET.SubElement(self.model.worldbody,'light',{'name':'preview_key','pos':'.5 -.5 2.7','dir':'-.15 .2 -1','diffuse':'.35 .35 .35','ambient':'.06 .06 .06','specular':'.05 .05 .05','castshadow':'true'})
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter=next(v for v in self.fixtures.values() if isinstance(v,Counter))
        self.init_robot_base_ref=self.counter
    def _create_objects(self):
        cube=BoxObject('cube',size=[.02]*3,rgba=[.8,.06,.04,1],density=self.initial_spec.get('density',800),friction=[self.initial_spec.get('friction',1),.005,.0001])
        self.objects={'cube':cube};self.model.merge_objects([cube])
        self.object_cfgs=[{'name':'cube','type':'object','init_robot_here':True,
                          'placement':{'fixture':self.counter,'size':(.1,.1),'pos':(-.18,-.6),'rotation':0},
                          'info':{'cat':'cube','groups_containing_sampled_obj':['cube'],'mjcf_path':'procedural_box'}}]
    def _check_success(self):
        if not hasattr(self,'cube_dof') or not hasattr(self,'obj_body_id') or 'cube' not in self.obj_body_id:return False
        p=self.sim.data.body_xpos[self.obj_body_id['cube']]
        return bool(np.linalg.norm(p[:2]-self.goal[:2])<.045 and abs(p[2]-self.goal[2])<.012
                    and self.check_contact(self.objects['cube'],self.counter)
                    and np.linalg.norm(p-self.sim.data.site_xpos[self.robots[0].eef_site_id['right']])>.10
                    and np.linalg.norm(self.sim.data.qvel[self.cube_dof:self.cube_dof+3])<.06)
    def get_ep_meta(self):
        m=super().get_ep_meta();m['lang']='Pick the red cube and release it on the green counter target.'
        m['custom_task']=True;m['layout_variant']='minimal procedural RoboCasa counter';return m

def make_env(width=640,height=480,seed=20261003,initial_spec=None):
    cfg=load_composite_controller_config(robot='PandaOmron')
    cfg['body_parts']['right'].update({'type':'OSC_POSE','input_type':'delta','input_ref_frame':'world','kp':200,'damping_ratio':1,'impedance_mode':'fixed','output_max':[.04,.04,.04,.25,.25,.25],'output_min':[-.04,-.04,-.04,-.25,-.25,-.25]})
    env=suite.make('CounterCubePreview',initial_spec=initial_spec,robots='PandaOmron',controller_configs=cfg,
                   layout_ids=[1],style_ids=[1],generative_textures=False,randomize_cameras=False,
                   use_distractors=False,clutter_mode=0,robot_spawn_deviation_pos_x=0,robot_spawn_deviation_pos_y=0,
                   robot_spawn_deviation_rot=0,initialization_noise=None,
                   has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,use_object_obs=True,
                   camera_names=['preview'],camera_widths=width,camera_heights=height,
                   render_gpu_device_id=0,control_freq=20,horizon=1500,ignore_done=True,seed=seed)
    env.reset();env.hard_reset=False
    from .initial_state import prepare_initial_state
    prepare_initial_state(env)
    return env
