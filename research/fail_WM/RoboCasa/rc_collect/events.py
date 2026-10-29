"""Labels use measured contacts and object dynamics, never branch names."""
import numpy as np
import mujoco
EVENT_NAMES=['contact','grasp','lift','drop','release','place','success']
class EventLabeler:
    def __init__(self,env):
        self.env=env;self.seen_lift=False;self.prev_grasp=False;self.prev_aperture=None
        self.cube_geoms={env.sim.model.geom_name2id(n) for n in env.objects['cube'].contact_geoms}
        r=env.robots[0]
        self.gripper_geoms={env.sim.model.geom_name2id(n) for n in r.gripper['right'].contact_geoms}
    def measure(self):
        e=self.env;d=e.sim.data;r=e.robots[0];p=d.body_xpos[e.obj_body_id['cube']]
        aperture=float(np.abs(d.qpos[r._ref_gripper_joint_pos_indexes['right']]).sum())
        contact=False;force=0.;pairs=[]
        for i in range(d.ncon):
            c=d.contact[i];a,b=int(c.geom1),int(c.geom2)
            if (a in self.cube_geoms and b in self.gripper_geoms) or (b in self.cube_geoms and a in self.gripper_geoms):
                contact=True;f=np.zeros(6);mujoco.mj_contactForce(e.sim.model._model,d._data,i,f);force+=abs(f[0]);pairs.append([a,b])
        grasp=bool(e._check_grasp(r.gripper['right'],e.objects['cube']))
        lift=bool(p[2]>.94+.06);self.seen_lift |= lift
        # Actual loss of support and downward velocity in the air, not a planned phase label.
        vz=float(d.qvel[e.cube_dof+2]);drop=bool(self.seen_lift and not grasp and p[2]>.98 and vz<-.15)
        release=bool(self.prev_grasp and not grasp and self.prev_aperture is not None and aperture>self.prev_aperture+.001)
        place=bool(e.check_contact(e.objects['cube'],e.counter) and np.linalg.norm(p[:2]-e.goal[:2])<.045)
        self.prev_grasp=grasp;self.prev_aperture=aperture
        return np.array([contact,grasp,lift,drop,release,place,e._check_success()],dtype=np.uint8),force,pairs
