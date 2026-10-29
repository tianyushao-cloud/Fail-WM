import numpy as np
from .action_conversion import native_action
# Grip-site +z points down; finger closing direction approximately world x.
DOWN=np.diag([-1.,1.,-1.])
class ScriptedController:
    def __init__(self,env,step=None):self.env=env;self.step_fn=step or (lambda a,p:env.step(a))
    def cube(self):return self.env.sim.data.body_xpos[self.env.obj_body_id['cube']].copy()
    def eef(self):return self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id['right']].copy()
    def move(self,p,gripper=-1,phase='move',max_steps=110,min_steps=12):
        p=np.asarray(p);stable=0
        for i in range(max_steps):
            self.step_fn(native_action(self.env,p,DOWN,gripper),phase)
            stable=stable+1 if np.linalg.norm(self.eef()-p)<.007 else 0
            if i>=min_steps and stable>=5:break
        print(phase,'steps',i+1,'eef',np.round(self.eef(),3),'cube',np.round(self.cube(),3),flush=True)
    def hold(self,p,g,steps,phase):
        for _ in range(steps):self.step_fn(native_action(self.env,p,DOWN,g),phase)
    def attempt(self,offset=(0,0,0),early_close=False):
        p=self.cube()+np.asarray(offset)
        self.move([p[0],p[1],1.16],-1,'approach',max_steps=160)
        if early_close:self.hold(self.eef(),1,16,'early_close')
        self.move(p,1 if early_close else -1,'descend')
        self.hold(p,1,18,'close')
        self.move([p[0],p[1],1.18],1,'lift')
    def place(self,drop=False):
        goal=self.env.goal
        if drop:
            self.move([.60,-.18,1.12],1,'transport_before_drop')
            self.hold(self.eef(),-1,30,'early_release')
        self.move([goal[0],goal[1],1.18],-1 if drop else 1,'transport')
        self.move(goal+[0,0,.005],-1 if drop else 1,'lower')
        self.hold(goal+[0,0,.005],-1,16,'release')
        self.move(goal+[0,0,.18],-1,'retreat')
        self.hold(self.eef(),-1,25,'settle')
    def run(self,kind):
        self.attempt(offset=(-.12,0,0) if kind in ('offset_miss','recovery') else (0,0,0),early_close=kind=='early_close')
        if kind=='recovery':
            from .recovery import retry_after_measured_failure
            retry_after_measured_failure(self)
        self.place(drop=kind=='early_release')
