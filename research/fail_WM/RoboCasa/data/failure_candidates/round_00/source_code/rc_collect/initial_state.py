import numpy as np

def prepare_initial_state(env):
    from robocasa.utils.env_utils import set_robot_to_position
    set_robot_to_position(env,np.array([.30,-.60,0.]));env.init_robot_base_pos=np.array([.30,-.60,0.])
    r=env.robots[0]
    env.sim.data.qpos[r._ref_joint_pos_indexes]=r.init_qpos
    env.sim.data.qpos[r._ref_gripper_joint_pos_indexes['right']]=[.04,-.04]
    env.sim.data.qvel[:]=0
    env.sim.forward()
    for c in r.part_controllers.values():
        c.update(force=True);c.reset_goal()
    r.gripper['right'].current_action=np.array([1.,-1.])
    cube=env.objects['cube'];env.cube_joint=env.sim.model.joint_name2id(cube.joints[0]);env.cube_dof=int(env.sim.model.jnt_dofadr[env.cube_joint])
    # Place the initial cube once before recording, identical across all branches.
    env.sim.data.set_joint_qpos(cube.joints[0],np.r_[env.initial_spec.get('cube_position',[.60,-.18,.941]),[np.cos(env.initial_spec.get('cube_yaw',0)/2),0,0,np.sin(env.initial_spec.get('cube_yaw',0)/2)]])
    env.sim.data.qvel[env.cube_dof:env.cube_dof+6]=0;env.sim.forward()
