"""Final finite-values, geometry consistency and package-level completeness audit."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,h5py,mujoco

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/batch1000');ap.add_argument('--expected',type=int,default=1000);ap.add_argument('--expected-failures',type=int);a=ap.parse_args();root=Path(a.data)
    folders=sorted(p for p in (root/'episodes').iterdir() if (p/'metadata.json').exists());assert len(folders)==a.expected
    plan=json.loads((root/'plan.json').read_text());planned={j['episode_id'] for j in plan['episodes']};assert {p.name for p in folders}==planned
    groups=defaultdict(set);group_outcomes=defaultdict(Counter);cats=Counter();splits=Counter();versions=Counter();pose_max=0.;physical_drift=0.;repaired=0;rows=[]
    geometry_checks=set(np.linspace(0,len(folders)-1,30,dtype=int))
    for j,p in enumerate(folders):
        meta=json.loads((p/'metadata.json').read_text());v=meta['validation'];assert meta['complete'] and v['sampling_sync_pass'] and v['replay_pass'] and v['alignment_pass']
        groups[meta['initial_state_id']].add(meta['split']);group_outcomes[meta['initial_state_id']]['success' if meta['success'] else 'failure']+=1;cats[meta['category']]+=1;splits[meta['split']]+=1;versions[meta['action_generation_version']]+=1
        with h5py.File(p/'trajectory.h5') as f:
            assert f['actions/raw'].shape==(meta['actions'],12)
            assert f['actions/target13'].shape==(meta['actions'],13)
            assert f['observations/eef_executed13'].shape==(meta['actions']+1,13)
            assert bool(f['events/flags'][-1,-1])==bool(meta['success'])
            bad=[]
            def finite(k,d):
                if isinstance(d,h5py.Dataset) and d.dtype.kind=='f' and not np.isfinite(d[:]).all():bad.append(k)
            f.visititems(finite);assert not bad,(p,bad)
            assert 'mj_forward at current qpos' in f.attrs['observation_sampling']
            if meta.get('observation_resampled_from_native_replay'):
                repaired+=1;assert hashlib.sha256(f['actions/raw'][:].tobytes()).hexdigest()==meta['raw_action_sha256']
                err=meta['sampling_fix_original_replay_errors'];assert err['raw_actions_byte_identical'];physical_drift=max(physical_drift,err['qpos'],err['qvel'],err['target13'])
            if j in geometry_checks:
                # Independent model/data, not recorder/replay environment or its caches.
                m=mujoco.MjModel.from_xml_string((p/'model.xml').read_text());d=mujoco.MjData(m)
                sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_right_grip_site');bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'cube_main')
                for i in range(len(f['observations/qpos'])):
                    d.qpos[:]=f['observations/qpos'][i];mujoco.mj_kinematics(m,d)
                    tcp=np.r_[d.site_xpos[sid],d.site_xmat[sid].reshape(9)];obj=np.r_[d.xpos[bid],d.xquat[bid]]
                    pose_max=max(pose_max,float(np.max(abs(tcp-f['observations/eef_executed13'][i,:12]))),float(np.max(abs(obj-f['observations/object_pose_wxyz'][i]))))
        rows.append({'episode_id':p.name,'validation_pass':True})
    assert all(len(s)==1 for s in groups.values())
    if a.expected_failures is None:assert dict(cats)=={'reference':300,'single':500,'double':100,'recovery':100}
    else:
        assert sum(c['failure'] for c in group_outcomes.values())==a.expected_failures
        assert all(c=={'success':3,'failure':7} for c in group_outcomes.values())
        selection=json.loads((root/'selection_manifest.json').read_text());assert len(selection['records'])==a.expected
        for row in selection['records']:
            src=Path(row['source_path']);dst=root/'episodes'/row['episode_id']
            meta=json.loads((dst/'metadata.json').read_text());assert meta['success']==row['success'] and meta['collection_plan_hash']==row['source_collection_plan_hash']
            for filename in ['trajectory.h5','rgb.mp4','metadata.json','model.xml','validation.json']:
                assert (src/filename).stat().st_ino==(dst/filename).stat().st_ino
        assert cats['recovery']==100
    assert dict(splits)=={'train':800,'val':100,'test':100}
    assert pose_max<1e-6 and physical_drift<1e-6
    report={'episodes':len(folders),'all_pass':True,'all_float_fields_finite':True,'all_action_state_rgb_alignment_pass':True,'all_native_action_replay_pass':True,'all_current_qpos_sampling_sync_pass':True,
      'initial_state_groups':len(groups),'split_group_overlap':0,'categories':dict(cats),'splits':dict(splits),'action_generation_versions':dict(versions),
      'native_action_preserving_resampled_episodes':repaired,'resampling_max_original_trajectory_drift':physical_drift,'independent_geometry_audit_episodes':len(geometry_checks),
      'independent_geometry_audit_all_timesteps':True,'independent_geometry_max_error':pose_max,
      'successes':sum(c['success'] for c in group_outcomes.values()),'failures':sum(c['failure'] for c in group_outcomes.values()),
      'outcome_conditioned_selection':a.expected_failures is not None,'original_source_file_identity_pass':a.expected_failures is not None,
      'final_success_matches_simulator_event_state':True,'native_action_dimension':12,'target_and_executed_representation_dimension':13}
    (root/'final_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
