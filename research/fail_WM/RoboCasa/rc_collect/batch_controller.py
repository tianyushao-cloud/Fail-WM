import numpy as np
from scipy.spatial.transform import Rotation
from .action_conversion import native_action
from .controller import DOWN

class BatchController:
    def __init__(self,env,recorder,job):
        self.e=env;self.rec=recorder;self.job=job;self.f={f['type']:f for f in job['perturbation']['factors']};self.rng=np.random.default_rng(job['action_seed'])
        self.R=Rotation.from_euler('z',env.initial_spec['cube_yaw']).as_matrix()@DOWN
        self.profile=job['reference_profile'];self.attempt_index=0;self.recovery_attempted=False;self.timed_out_phases=[]
    def cube(self):return self.e.sim.data.body_xpos[self.e.obj_body_id['cube']].copy()
    def eef(self):return self.e.sim.data.site_xpos[self.e.robots[0].eef_site_id['right']].copy()
    def move(self,p,g=-1,phase='move',rotation=None,max_steps=85,min_steps=None):
        p=np.asarray(p,float);R=self.R if rotation is None else rotation;stable=0;phase=f'attempt{self.attempt_index}/{phase}'
        if min_steps is None:min_steps=[12,16,20][self.profile]
        for i in range(max_steps):
            self.rec.step(native_action(self.e,p,R,g),phase)
            stable=stable+1 if np.linalg.norm(self.eef()-p)<.007 else 0
            if i>=min_steps and stable>=5:break
        if stable<5:self.timed_out_phases.append({'phase':phase,'position_error_m':float(np.linalg.norm(self.eef()-p)),'steps':i+1})
    def hold(self,p,g,steps,phase,rotation=None):
        R=self.R if rotation is None else rotation
        for _ in range(steps):self.rec.step(native_action(self.e,p,R,g),f'attempt{self.attempt_index}/{phase}')
    def push(self,f):
        p=self.cube();axis=f['axis'];direction=np.zeros(3);direction[axis]=f['sign']
        start=p-direction*.09;end=p+direction*f['sweep_distance_m']
        self.move(start+[0,0,.15],-1,'push_approach');self.hold(self.eef(),1,14,'push_preclose')
        self.move(start+[0,0,.008],1,'push_lower');self.move(end+[0,0,.008],1,'push_sweep',max_steps=65)
        self.move(self.eef()+[0,0,.16],-1,'push_retreat');self.hold(self.eef(),-1,20,'push_settle')
        return p
    def attempt(self,factors=None,cached=None):
        factors=self.f if factors is None else factors
        p=self.cube() if cached is None else np.asarray(cached).copy();R=self.R
        if 'grasp_offset' in factors:p+=np.asarray(factors['grasp_offset']['xyz_m'])
        if 'orientation_tilt' in factors:
            f=factors['orientation_tilt'];rv=np.zeros(3);rv[f['axis']]=f['angle_rad'];R=Rotation.from_rotvec(rv).as_matrix()@R
        self.move([p[0],p[1],[1.16,1.12,1.20][self.profile]],-1,'approach',rotation=R,max_steps=100)
        early='early_close' in factors
        if early:self.hold(self.eef(),1,factors['early_close']['close_steps_before_descend'],'early_close',rotation=R)
        if 'skip_descend' in factors:
            self.hold(self.eef(),-1,factors['skip_descend']['pause_steps'],'pause_skip_descend',rotation=R);p=self.eef()
        else:self.move(p,1 if early else -1,'descend',rotation=R)
        if 'delayed_close' in factors:
            p=p+[0,0,factors['delayed_close']['open_lift_distance_m']];self.move(p,-1,'delay_before_close',rotation=R)
        if 'partial_close' in factors:
            self.hold(p,1,factors['partial_close']['positive_close_action_steps'],'partial_close',rotation=R);self.hold(p,0,18,'partial_hold',rotation=R);g=0
        else:self.hold(p,1,[18,22,26][self.profile],'close',rotation=R);g=1
        self.move([p[0],p[1],1.18],g,'lift',rotation=R)
        return g
    def grasped_lifted(self):
        return bool(self.e._check_grasp(self.e.robots[0].gripper['right'],self.e.objects['cube']) and self.cube()[2]>1.)
    def drop(self,f,g):
        obj=self.cube();xy=obj[:2]+f['transport_fraction']*(self.e.goal[:2]-obj[:2]);p=np.r_[xy,f['release_height_m']]
        self.move(p,g,'transport_before_drop');self.hold(p,-1,30,'early_release')
    def place(self,g,factors=None):
        factors=self.f if factors is None else factors;goal=self.e.goal.copy();start=self.cube()
        if 'place_offset' in factors:goal+=np.asarray(factors['place_offset']['xyz_m'])
        if 'transport_short' in factors:goal[:2]=start[:2]+factors['transport_short']['transport_fraction']*(goal[:2]-start[:2])
        if 'waypoint' in factors:
            waypoint=(self.eef()+np.r_[goal[:2],1.18])/2+np.asarray(factors['waypoint']['xyz_offset_m']);waypoint[2]=max(waypoint[2],.96)
            self.move(waypoint,g,'modified_waypoint')
        elif self.profile==2:self.move(np.r_[(start[:2]+goal[:2])/2+[0,-.035],1.20],g,'reference_arc')
        self.move(np.r_[goal[:2],1.18],g,'transport');self.move(goal+[0,0,.005],g,'lower')
        self.hold(goal+[0,0,.005],-1,16,'release');self.move(goal+[0,0,.18],-1,'retreat');self.hold(self.eef(),-1,25,'settle')
    def run(self):
        cached=self.push(self.f['push']) if 'push' in self.f else None
        g=self.attempt(cached=cached)
        if 'early_release' in self.f:self.drop(self.f['early_release'],g);g=-1
        if self.job['category']=='recovery' and not self.grasped_lifted():
            self.recovery_attempted=True;self.hold(self.eef(),-1,25,'recovery_wait');self.attempt_index=1
            p=self.cube()
            if p[2]>.85 and abs(p[0])<.77 and abs(p[1])<.34:
                # Match the *current* measured cube yaw; no reset between attempts.
                quat=self.e.sim.data.body_xquat[self.e.obj_body_id['cube']];yaw=Rotation.from_quat(np.r_[quat[1:],quat[0]]).as_euler('xyz')[2]
                self.R=Rotation.from_euler('z',yaw).as_matrix()@DOWN;g=self.attempt(factors={});self.place(g,factors={})
            else:
                self.move(np.r_[np.clip(p[:2],[.40,-.28],[.68,-.07]),1.14],-1,'recovery_object_out_of_reach')
                self.hold(self.eef(),-1,30,'recovery_observe')
        else:self.place(g)
        return {'recovery_attempted':self.recovery_attempted,'controller_phase_timeouts':self.timed_out_phases}
