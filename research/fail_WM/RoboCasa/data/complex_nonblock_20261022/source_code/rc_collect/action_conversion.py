"""World-frame pose, rotation matrix row-major; +1=open, -1=closed."""
import numpy as np
from scipy.spatial.transform import Rotation

def pose13(position, rotation, gripper):
    return np.r_[position, np.asarray(rotation).reshape(9), float(gripper)]

def executed13(env):
    r=env.robots[0];d=env.sim.data;sid=r.eef_site_id['right']
    q=d.qpos[r._ref_gripper_joint_pos_indexes['right']]
    aperture=float(np.sum(np.abs(q)))
    return pose13(d.site_xpos[sid],d.site_xmat[sid].reshape(3,3),np.clip(2*aperture/.08-1,-1,1))

def controller_target13(env):
    r=env.robots[0];c=r.part_controllers['right']
    return pose13(c.goal_pos,c.goal_ori,r.gripper['right'].current_action[0])

def native_action(env, position, rotation, gripper):
    r=env.robots[0];sid=r.eef_site_id['right'];d=env.sim.data
    a=np.zeros(env.action_dim);a[-1]=-1  # hybrid controller arm mode
    lo,hi=r._action_split_indexes['right']
    delta=np.r_[(np.asarray(position)-d.site_xpos[sid])/.04,
       Rotation.from_matrix(np.asarray(rotation)@d.site_xmat[sid].reshape(3,3).T).as_rotvec()/.25]
    a[lo:hi]=np.clip(delta,-1,1)
    lo,hi=r._action_split_indexes['right_gripper'];a[lo:hi]=gripper
    return a
