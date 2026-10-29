"""Four independent spawn workers; atomic episodes, retries, deterministic groups."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
import argparse,hashlib,json,time,traceback,signal,shutil,multiprocessing as mp
from pathlib import Path
from collections import Counter
from rc_collect.batch_plan import make_plan,plan_hash

def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def worker(item):
    job,root_str,hashcode=item;root=Path(root_str);final=root/'episodes'/job['episode_id'];start=time.perf_counter()
    from rc_collect.environment import make_env
    from rc_collect.recording import Recorder
    from rc_collect.batch_controller import BatchController
    from rc_collect.batch_validation import validate_episode
    def timeout(signum,frame):raise TimeoutError('episode exceeded 240 seconds')
    signal.signal(signal.SIGALRM,timeout)
    for attempt in range(3):
        e=None;tmp=root/'incomplete'/f'{job["episode_id"]}_pid{os.getpid()}_try{attempt}';rec=None
        try:
            signal.alarm(240)
            e=make_env(seed=job['seed'],initial_spec=job['initial_spec'])
            info={k:job[k] for k in ['episode_id','initial_state_id','seed','action_seed','split','category','group_index','branch_index','reference_profile','initial_spec']}
            info.update(collection_plan_hash=hashcode,schema_version='2.1',action_generation_version='2.1_synchronized_feedback',worker_pid=os.getpid())
            rec=Recorder(e,tmp,job['category'],episode_info=info);controller=BatchController(e,rec,job)
            control_info=controller.run();meta=rec.finish(job['perturbation']);meta.update(control_info)
            meta['physical_parameters']={'cube_mass_kg':float(e.sim.model.body_mass[e.obj_body_id['cube']]),'density_kg_m3':job['initial_spec']['density'],'sliding_friction':job['initial_spec']['friction']}
            validation=validate_episode(tmp,e);meta['validation']=validation;meta['complete']=True;meta['wall_seconds']=time.perf_counter()-start
            atomic_json(tmp/'metadata.json',meta);atomic_json(tmp/'validation.json',validation)
            final.parent.mkdir(exist_ok=True,parents=True)
            if final.exists():raise FileExistsError(final)
            tmp.replace(final);signal.alarm(0)
            return {'ok':True,'metadata':meta}
        except Exception:
            signal.alarm(0);error=traceback.format_exc();(root/'errors').mkdir(exist_ok=True)
            atomic_json(root/'errors'/f'{job["episode_id"]}_try{attempt}.json',{'job':job,'traceback':error})
            if rec is not None:
                try:rec.writer.close()
                except Exception:pass
            if attempt==2:return {'ok':False,'episode_id':job['episode_id'],'error':error}
        finally:
            if e is not None:
                try:e.close()
                except Exception:pass

def summarize(root,jobs,rows,status,start,workers,errors):
    splitgroups={s:sorted({r['initial_state_id'] for r in rows if r['split']==s}) for s in ['train','val','test']}
    recovery=[r for r in rows if r['category']=='recovery']
    report={'status':status,'requested_episodes':len(jobs),'completed':len(rows),'workers':workers,'wall_seconds_this_run':round(time.perf_counter()-start,2),
      'successes':sum(r['success'] for r in rows),'failures':sum(not r['success'] for r in rows),'actual_failure_types':dict(Counter(r['failure_type'] for r in rows if not r['success'])),
      'completed_by_category':dict(Counter(r['category'] for r in rows)),'perturbation_counts':dict(Counter(f['type'] for r in rows for f in r['perturbation']['factors'])),
      'recovery_attempted':sum(r.get('recovery_attempted',False) for r in recovery),'recovery_successes':sum(r.get('recovery_attempted',False) and r['success'] for r in recovery),
      'replay_passed':sum(r['validation'].get('replay_pass',False) for r in rows),'alignment_passed':sum(r['validation']['alignment_pass'] for r in rows),
      'total_actions':sum(r['actions'] for r in rows),'total_rgb_frames':sum(r['frames'] for r in rows),'groups_by_split':{s:len(g) for s,g in splitgroups.items()},
      'split_group_overlap':len(set(splitgroups['train'])&set(splitgroups['val']))+len(set(splitgroups['train'])&set(splitgroups['test']))+len(set(splitgroups['val'])&set(splitgroups['test'])),'unresolved_errors':errors}
    atomic_json(root/'progress.json',report)
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='data/batch1000');ap.add_argument('--episodes',type=int,default=1000);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--seed',type=int,default=2026100301);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--plan-file',type=Path);args=ap.parse_args()
    root=Path(args.output).resolve();root.mkdir(exist_ok=True,parents=True)
    import fcntl
    lock=(root/'.coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=json.loads(args.plan_file.read_text())['episodes'] if args.plan_file else make_plan(args.episodes,args.seed)
    assert jobs and len({j['episode_id'] for j in jobs})==len(jobs)
    h=plan_hash(jobs);plan=root/'plan.json'
    if plan.exists():
        old=json.loads(plan.read_text());assert old['plan_hash']==h,'existing output has a different plan'
    else:atomic_json(plan,{'plan_hash':h,'seed':args.seed,'workers':args.workers,'episodes':jobs})
    # Preserve code used to collect this dataset, independent of later edits.
    source=root/'source_code'
    if not source.exists():
        source.mkdir();shutil.copy2(__file__,source/'batch_collect.py');shutil.copytree(Path(__file__).parent/'rc_collect',source/'rc_collect',ignore=shutil.ignore_patterns('__pycache__'))
    rows=[];pending=[];errors=[];start=time.perf_counter()
    for job in jobs:
        p=root/'episodes'/job['episode_id']/'metadata.json'
        if p.exists():
            m=json.loads(p.read_text());assert m.get('complete') and m['collection_plan_hash']==h and m['validation']['replay_pass'] and m['validation'].get('sampling_sync_pass');rows.append(m)
        else:pending.append(job)
    if args.limit:pending=pending[:max(0,args.limit-len(rows))]
    summarize(root,jobs,rows,'running',start,args.workers,errors)
    with mp.get_context('spawn').Pool(args.workers,maxtasksperchild=25) as pool:
        for result in pool.imap_unordered(worker,[(j,str(root),h) for j in pending],chunksize=1):
            if result['ok']:rows.append(result['metadata'])
            else:errors.append(result)
            report=summarize(root,jobs,rows,'running',start,args.workers,errors)
            if (len(rows)+len(errors))%10==0:print(json.dumps({k:report[k] for k in ['completed','successes','failures','actual_failure_types','wall_seconds_this_run','replay_passed']}),flush=True)
    status='complete' if len(rows)==len(jobs) and not errors else 'pilot_complete' if args.limit and not errors else 'incomplete'
    report=summarize(root,jobs,rows,status,start,args.workers,errors);atomic_json(root/'statistics.json',report)
    rows.sort(key=lambda r:r['episode_id']);(root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    atomic_json(root/'splits.json',{s:[r['episode_id'] for r in rows if r['split']==s] for s in ['train','val','test']})
    print(json.dumps(report,indent=2),flush=True)
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
