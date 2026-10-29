"""Refresh MuJoCo derived geometry at qpos(t), without advancing physics.

mj_step2 integrates qpos but leaves transforms at the previous 2ms substep.
mj_forward refreshes them and also changes solver warmstart. Restore all
integration inputs afterwards so reading an observation never changes physics.
"""
import mujoco,numpy as np

def synchronize_observation(env):
    m=env.sim.model._model;d=env.sim.data._data;spec=int(mujoco.mjtState.mjSTATE_INTEGRATION)
    before=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,before,spec)
    env.sim.forward()
    mujoco.mj_setState(m,d,before,spec)
    return before
