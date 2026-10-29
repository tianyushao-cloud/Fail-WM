"""Real-physics supplementary plans, then explicit outcome-conditioned selection.

Exactly 3 successful and 7 failed episodes per pre-existing initial group.
No outcome relabeling, state edits, video edits, or discarded candidate evidence.
"""
import argparse, json, hashlib, os, shutil
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

MODES=['early_release','place_offset','push','partial_close','transport_short','grasp_offset','early_close','delayed_close','skip_descend','orientation_tilt','waypoint']
INTENDED={'early_release':'dropped_before_goal','place_offset':'lifted_but_not_placed','push':'object_off_counter','partial_close':'contact_without_lift','transport_short':'lifted_but_not_placed','grasp_offset':'grasp_miss','early_close':'contact_without_lift','delayed_close':'grasp_miss','skip_descend':'grasp_miss','orientation_tilt':'contact_without_lift','waypoint':'dropped_before_goal'}

def factor(mode,rng,spec):
    u=lambda a,b:float(rng.uniform(a,b))
    if mode=='early_release':return {'type':mode,'transport_fraction':u(0,.08),'release_height_m':u(1.14,1.20)}
    if mode=='place_offset':
        # Point toward center so wrong-place examples often remain on the counter.
        d=np.array([.56,-.18])-np.array(spec['goal_position'][:2]);d/=np.linalg.norm(d)
        return {'type':mode,'xyz_m':[*list(d*u(.085,.12)),0.]}
    if mode=='push':return {'type':mode,'axis':1,'sign':-1,'sweep_distance_m':u(.22,.32)}
    if mode=='partial_close':return {'type':mode,'positive_close_action_steps':int(rng.integers(1,3))}
    if mode=='transport_short':return {'type':mode,'transport_fraction':u(.04,.22)}
    if mode=='grasp_offset':return {'type':mode,'xyz_m':[-u(.085,.13),u(-.015,.015),u(-.005,.01)]}
    if mode=='early_close':return {'type':mode,'close_steps_before_descend':int(rng.integers(13,23))}
    if mode=='delayed_close':return {'type':mode,'open_lift_distance_m':u(.085,.15)}
    if mode=='skip_descend':return {'type':mode,'pause_steps':int(rng.integers(4,31))}
    if mode=='orientation_tilt':return {'type':mode,'axis':int(rng.integers(0,2)),'angle_rad':u(.80,1.12)*int(rng.choice([-1,1]))}
    if mode=='waypoint':return {'type':mode,'xyz_offset_m':[u(-.07,.07),u(-.06,.06),u(-.22,-.18)]}
    raise ValueError(mode)

def load_pool(base,candidates):
    pool={}
    for root in [base,*sorted(candidates.glob('round_*'))]:
        for p in sorted((root/'episodes').glob('*/metadata.json')):
            m=json.loads(p.read_text());v=m['validation']
            assert m['complete'] and v['replay_pass'] and v['alignment_pass'] and v['sampling_sync_pass']
            assert m['episode_id'] not in pool
            pool[m['episode_id']]=(p.parent,m)
    bygroup=defaultdict(list)
    for p,m in pool.values():bygroup[m['initial_state_id']].append((p,m))
    return pool,bygroup

def plan(base,candidates,round_index):
    original=json.loads((base/'plan.json').read_text())['episodes'];initials={j['initial_state_id']:j for j in original if j['branch_index']==0}
    pool,groups=load_pool(base,candidates);jobs=[];needs={}
    for gid,template in sorted(initials.items()):
        failures=[m for p,m in groups[gid] if not m['success']];deficit=max(0,7-len(failures))
        if not deficit:continue
        needs[gid]=deficit;counts=Counter(m['failure_type'] for m in failures)
        # Prefer failure modes currently underrepresented within each initial group.
        modes=MODES if round_index==0 else ['early_release','place_offset','partial_close','grasp_offset','transport_short','skip_descend','delayed_close','early_close','push']
        if round_index>=2:modes=['partial_close','transport_short','skip_descend','grasp_offset']
        ranked=sorted(modes,key=lambda mode:(counts[INTENDED[mode]],(MODES.index(mode)+template['group_index'])%len(MODES)))
        for k in range(deficit+1):
            branch=10+round_index*20+k;seed=2026300301+template['group_index']*100+branch
            rng=np.random.default_rng(seed);mode=ranked[k%len(ranked)];spec=dict(template['initial_spec']);fs=[factor(mode,rng,spec)]
            category='single'
            if k==deficit and template['group_index']%3==0:
                second='early_close' if mode in ['grasp_offset','orientation_tilt'] else 'orientation_tilt'
                fs.append(factor(second,rng,spec));category='double'
            j={**template,'episode_id':f'g{template["group_index"]:05d}_b{branch:02d}','branch_index':branch,'action_seed':seed,'reference_profile':0,'category':category,'initial_spec':spec,
               'perturbation':{'factors':fs,'recovery_requested':False,'parameters_are_action_intents_not_outcome_labels':True}}
            jobs.append(j)
    candidates.mkdir(exist_ok=True,parents=True)
    out={'episodes':jobs,'round':round_index,'existing_validated_pool':len(pool),'failed_episode_deficit_by_group':needs,'target_final_failures':700,'target_final_successes':300,'selection_is_outcome_conditioned':True}
    p=candidates/f'round_{round_index:02d}_plan.json';p.write_text(json.dumps(out,indent=2))
    print(json.dumps({'plan_file':str(p),'candidate_jobs':len(jobs),'missing_failures':sum(needs.values()),'groups_needing_candidates':len(needs)}))

def assemble(base,candidates,output):
    pool,groups=load_pool(base,candidates);assert len(groups)==100
    selected=[];short={};failure_counter=Counter();success_factor_counter=Counter()
    for gid,entries in sorted(groups.items()):
        successes=[x for x in entries if x[1]['success']];failures=[x for x in entries if not x[1]['success']]
        if len(successes)<3 or len(failures)<7:short[gid]={'successes':len(successes),'failures':len(failures)};continue
        recovery=[x for x in entries if x[1]['category']=='recovery'];assert len(recovery)==1
        # Keep all 100 original recovery episodes, including the seven failed ones.
        successful_recovery=[x for x in recovery if x[1]['success']]
        references=sorted([x for x in successes if x[1]['category']=='reference'],key=lambda x:x[1]['episode_id'])
        chosen_success=successful_recovery+[references[entries[0][1]['group_index']%len(references)]]
        perturbed_success=[x for x in successes if x[1]['category'] in ['single','double']]
        while len(chosen_success)<3 and perturbed_success:
            def successful_perturbation_rank(x):
                names=[f['type'] for f in x[1]['perturbation']['factors']]
                return (all(n in ['mass','friction'] for n in names),sum(success_factor_counter[n] for n in names),hashlib.sha256(('success:'+x[1]['episode_id']).encode()).hexdigest())
            x=min(perturbed_success,key=successful_perturbation_rank);perturbed_success.remove(x);chosen_success.append(x)
            success_factor_counter.update(f['type'] for f in x[1]['perturbation']['factors'])
        chosen_success.extend([x for x in references if x not in chosen_success][:3-len(chosen_success)])
        chosen_failure=[x for x in recovery if not x[1]['success']]
        local=Counter(x[1]['failure_type'] for x in chosen_failure)
        remaining=[x for x in failures if x not in chosen_failure]
        while len(chosen_failure)<7:
            # Diversity within each group, then across dataset; seeded hash breaks ties.
            x=min(remaining,key=lambda x:(local[x[1]['failure_type']],failure_counter[x[1]['failure_type']],hashlib.sha256(('20261004:'+x[1]['episode_id']).encode()).hexdigest()))
            remaining.remove(x);chosen_failure.append(x);local[x[1]['failure_type']]+=1
        failure_counter.update(local);selected.extend(chosen_success+chosen_failure)
    if short:
        print(json.dumps({'ready':False,'short_groups':short}));return
    assert len(selected)==1000 and sum(not m['success'] for p,m in selected)==700
    output.mkdir(exist_ok=True,parents=True);episodes=output/'episodes';episodes.mkdir(exist_ok=True)
    records=[];jobs=[]
    alljobs={}
    for root in [base,*sorted(candidates.glob('round_*'))]:
        if (root/'plan.json').exists():
            for j in json.loads((root/'plan.json').read_text())['episodes']:alljobs[j['episode_id']]=j
    for p,m in sorted(selected,key=lambda x:x[1]['episode_id']):
        dest=episodes/m['episode_id'];dest.mkdir(exist_ok=True)
        for src in p.iterdir():
            if src.is_file():
                target=dest/src.name
                if not target.exists():os.link(src,target)
                assert target.stat().st_ino==src.stat().st_ino
        records.append({'episode_id':m['episode_id'],'source_path':str(p),'success':m['success'],'failure_type':m['failure_type'],'initial_state_id':m['initial_state_id'],'split':m['split'],'source_collection_plan_hash':m['collection_plan_hash']})
        jobs.append(alljobs[m['episode_id']])
    policy={'method':'outcome-conditioned stratified subsampling of validated real-physics episodes','requested_failure_ratio':.7,'successes':300,'failures':700,'episodes':1000,
      'per_initial_group':{'success':3,'failure':7},'preserve_all_original_recovery_episodes':True,'candidate_pool_episodes':len(pool),
      'candidate_pool_successes':sum(m['success'] for p,m in pool.values()),'candidate_pool_failures':sum(not m['success'] for p,m in pool.values()),
      'labels_unchanged':True,'actions_states_videos_unchanged':True,'failure_type_diversity_priority':True,
      'successful_perturbations_preserved':True,'at_least_one_reference_success_per_initial_group':True,
      'selection_bias':'Selected dataset ratios are deliberately controlled and are not unbiased estimates of policy success rates. All unselected candidate outcomes remain available.',
      'records':records}
    (output/'selection_manifest.json').write_text(json.dumps(policy,indent=2))
    selected_ids={m['episode_id'] for p,m in selected}
    (output/'candidate_pool_manifest.jsonl').write_text(''.join(json.dumps({'episode_id':m['episode_id'],'source_path':str(p),'initial_state_id':m['initial_state_id'],'split':m['split'],'success':m['success'],'failure_type':m['failure_type'],'category':m['category'],'selected':m['episode_id'] in selected_ids,'perturbation':m['perturbation']})+'\n' for p,m in sorted(pool.values(),key=lambda x:x[1]['episode_id'])))
    (output/'plan.json').write_text(json.dumps({'episodes':jobs,'plan_type':'post-execution-selected-membership','original_collection_plans_preserved_in_metadata':True},indent=2))
    rows=[m for p,m in selected];(output/'manifest.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in sorted(rows,key=lambda m:m['episode_id'])))
    (output/'splits.json').write_text(json.dumps({s:sorted(m['episode_id'] for m in rows if m['split']==s) for s in ['train','val','test']},indent=2))
    sources=output/'source_code';sources.mkdir(exist_ok=True)
    if not (sources/'original_collection').exists():shutil.copytree(base/'source_code',sources/'original_collection')
    for r in candidates.glob('round_*'):
        if (r/'source_code').exists() and not (sources/r.name).exists():shutil.copytree(r/'source_code',sources/r.name)
    shutil.copy2(__file__,sources/Path(__file__).name)
    print(json.dumps({'ready':True,'episodes':1000,'successes':300,'failures':700,'failure_types':dict(failure_counter),'pool_episodes':len(pool),'output':str(output)}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['plan','assemble']);ap.add_argument('--base',type=Path,default=Path('data/batch1000'));ap.add_argument('--candidates',type=Path,default=Path('data/failure_candidates'));ap.add_argument('--output',type=Path,default=Path('data/batch1000_failure70'));ap.add_argument('--round',type=int,default=0);a=ap.parse_args()
    if a.stage=='plan':plan(a.base.resolve(),a.candidates.resolve(),a.round)
    else:assemble(a.base.resolve(),a.candidates.resolve(),a.output.resolve())
if __name__=='__main__':main()
