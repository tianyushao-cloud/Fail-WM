"""Deterministic plans: 100 initial groups x 10 branches, split before execution."""
import hashlib,json
import numpy as np
FAMILIES=['grasp_offset','orientation_tilt','early_close','delayed_close','partial_close','early_release','place_offset','transport_short','skip_descend','push','waypoint','mass','friction']

def factor(name,rng):
    if name in ('grasp_offset','place_offset'):
        amp=float(rng.uniform(.008,.15 if name=='place_offset' else .13));theta=float(rng.uniform(-np.pi,np.pi))
        return {'type':name,'xyz_m':[amp*np.cos(theta),amp*np.sin(theta),float(rng.uniform(-.008,.035)) if name=='grasp_offset' else 0.]}
    if name=='orientation_tilt':return {'type':name,'axis':int(rng.integers(0,2)),'angle_rad':float(rng.uniform(.12,1.05)*rng.choice([-1,1]))}
    if name=='early_close':return {'type':name,'close_steps_before_descend':int(rng.integers(8,23))}
    if name=='delayed_close':return {'type':name,'open_lift_distance_m':float(rng.uniform(.025,.17))}
    if name=='partial_close':return {'type':name,'positive_close_action_steps':int(rng.integers(1,6))}
    if name=='early_release':return {'type':name,'transport_fraction':float(rng.uniform(.0,.60)),'release_height_m':float(rng.uniform(1.08,1.20))}
    if name=='transport_short':return {'type':name,'transport_fraction':float(rng.uniform(.08,.7))}
    if name=='skip_descend':return {'type':name,'pause_steps':int(rng.integers(0,31))}
    if name=='push':return {'type':name,'axis':int(rng.integers(0,2)),'sign':int(rng.choice([-1,1])),'sweep_distance_m':float(rng.uniform(.08,.23))}
    if name=='waypoint':return {'type':name,'xyz_offset_m':[float(rng.uniform(-.12,.12)),float(rng.uniform(-.10,.10)),float(rng.uniform(-.20,.06))]}
    if name=='mass':return {'type':name,'density_kg_m3':float(rng.uniform(400,2000))}
    if name=='friction':return {'type':name,'sliding_friction':float(rng.uniform(.08,2.0))}
    raise ValueError(name)

def make_plan(episodes=1000,seed=2026100301):
    if episodes%10:raise ValueError('episodes must be a multiple of 10')
    groups=episodes//10;rng=np.random.default_rng(seed);perm=rng.permutation(groups)
    train=set(perm[:int(groups*.8)]);val=set(perm[int(groups*.8):int(groups*.9)])
    jobs=[]
    for group in range(groups):
        group_seed=seed+10000+group;irng=np.random.default_rng(group_seed)
        cube=[float(irng.uniform(.54,.64)),float(irng.uniform(-.23,-.15)),.941]
        for _ in range(1000):
            goal=[float(irng.uniform(.43,.68)),float(irng.uniform(-.28,-.06)),.942]
            if .12<np.linalg.norm(np.array(goal[:2])-cube[:2])<.22:break
        else:raise ValueError('unable to sample separated goal')
        initial={'cube_position':cube,'cube_yaw':float(irng.uniform(-.30,.30)),'goal_position':goal,'density':800.,'friction':1.}
        split='train' if group in train else 'val' if group in val else 'test'
        for branch in range(10):
            action_seed=seed+100000+group*10+branch;arng=np.random.default_rng(action_seed);spec=dict(initial)
            if branch<3:
                category='reference';factors=[];profile=branch
            elif branch<8:
                category='single';factors=[factor(FAMILIES[(group*5+branch-3)%len(FAMILIES)],arng)];profile=0
            elif branch==8:
                category='double';choice=group%4
                names=[['grasp_offset','early_close'],['orientation_tilt','early_release'],['partial_close','transport_short'],['place_offset','waypoint']][choice]
                factors=[factor(n,arng) for n in names];profile=0
            else:
                category='recovery';name=['grasp_offset','early_close','early_release','push'][group%4];factors=[factor(name,arng)];profile=0
                if name=='grasp_offset':factors=[{'type':name,'xyz_m':[-float(arng.uniform(.09,.13)),0.,0.]}]
            for f in factors:
                if f['type']=='mass':spec['density']=f['density_kg_m3']
                if f['type']=='friction':spec['friction']=f['sliding_friction']
            jobs.append({'episode_id':f'g{group:05d}_b{branch:02d}','initial_state_id':f'counter_group_{group:05d}', 'group_index':group,'branch_index':branch,
                'seed':group_seed,'action_seed':action_seed,'split':split,'category':category,'reference_profile':profile,'initial_spec':spec,
                'perturbation':{'factors':factors,'recovery_requested':category=='recovery','parameters_are_action_intents_not_outcome_labels':True}})
    return jobs

def plan_hash(jobs):return hashlib.sha256(json.dumps(jobs,sort_keys=True).encode()).hexdigest()
