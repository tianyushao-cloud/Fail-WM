"""Dataset statistics and a compact inspection set selected by actual outcomes."""
import argparse,json,shutil,random
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import h5py
from PIL import Image,ImageDraw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/batch1000');ap.add_argument('--expected',type=int,default=1000);ap.add_argument('--allow-partial',action='store_true');a=ap.parse_args();root=Path(a.data)
    rows=[json.loads(p.read_text()) for p in sorted((root/'episodes').glob('*/metadata.json'))]
    assert rows
    if not a.allow_partial:assert len(rows)==a.expected,(len(rows),a.expected)
    assert all(r['complete'] and r['validation']['replay_pass'] and r['validation']['alignment_pass'] for r in rows)
    groups=defaultdict(set);bycat=defaultdict(list);byfactor=defaultdict(list);sameinitial=defaultdict(set)
    for r in rows:
        groups[r['initial_state_id']].add(r['split']);bycat[r['category']].append(r)
        for f in r['perturbation']['factors']:byfactor[f['type']].append(r)
        initial_key=json.dumps(r['initial_spec'],sort_keys=True);sameinitial[(r['initial_state_id'],initial_key)].add(r['initial_state_sha256'])
    assert all(len(s)==1 for s in groups.values()),'initial group leaked across splits'
    assert all(len(s)==1 for s in sameinitial.values()),'same spec has inconsistent recorded initial state'
    lengths=np.array([r['actions'] for r in rows]);qerr=max(r['validation']['replay_max_qpos_error'] for r in rows)
    statistics={'episodes':len(rows),'initial_state_groups':len(groups),'successes':sum(r['success'] for r in rows),'failures':sum(not r['success'] for r in rows),
      'success_rate':float(np.mean([r['success'] for r in rows])),'failure_types':dict(Counter(r['failure_type'] for r in rows if not r['success'])),
      'category_outcomes':{k:{'episodes':len(rs),'successes':sum(r['success'] for r in rs),'failures':sum(not r['success'] for r in rs)} for k,rs in bycat.items()},
      'perturbation_outcomes':{k:{'episodes':len(rs),'successes':sum(r['success'] for r in rs),'failures':sum(not r['success'] for r in rs)} for k,rs in byfactor.items()},
      'recovery':{'requested':len(bycat['recovery']),'attempted':sum(r['recovery_attempted'] for r in bycat['recovery']),'recovered_successfully':sum(r['recovery_attempted'] and r['success'] for r in bycat['recovery']),'failed_after_recovery':sum(r['recovery_attempted'] and not r['success'] for r in bycat['recovery'])},
      'action_lengths':{'min':int(lengths.min()),'median':float(np.median(lengths)),'mean':float(lengths.mean()),'max':int(lengths.max())},
      'total_actions':int(lengths.sum()),'total_rgb_frames':int(lengths.sum()+len(rows)),'simulated_transition_seconds':float(lengths.sum()/20),
      'event_episode_counts':dict(Counter(name for r in rows for name,yes in r['event_summary'].items() if yes)),
      'splits':{s:{'episodes':sum(r['split']==s for r in rows),'groups':sum(s in gs for gs in groups.values())} for s in ['train','val','test']},
      'split_group_overlap':0,'initial_state_hash_consistency_pass':True,'validated_episodes':len(rows),'replay_max_qpos_error':qerr,
      'replay_max_qvel_error':max(r['validation']['replay_max_qvel_error'] for r in rows),'replay_max_target13_error':max(r['validation']['replay_max_target13_error'] for r in rows),
      'replay_sampled_rgb_max_mae':max(r['validation']['replay_sampled_rgb_max_mae'] for r in rows),
      'episode_bytes':sum(p.stat().st_size for p in (root/'episodes').rglob('*') if p.is_file())}
    (root/'dataset_statistics.json').write_text(json.dumps(statistics,indent=2))
    inspection=root/'inspection';inspection.mkdir(exist_ok=True);rng=random.Random(919);selected=[];used=set()
    selections=[('normal_success',lambda r:r['category']=='reference' and r['success'],2),
      ('grasp_miss',lambda r:r['failure_type']=='grasp_miss',2),('contact_without_lift',lambda r:r['failure_type']=='contact_without_lift',2),
      ('drop',lambda r:r['failure_type']=='dropped_before_goal',2),('wrong_place',lambda r:r['failure_type']=='lifted_but_not_placed',2),
      ('off_counter',lambda r:r['failure_type']=='object_off_counter',2),('recovery_success',lambda r:r['recovery_attempted'] and r['success'],2),
      ('recovery_failure',lambda r:r['recovery_attempted'] and not r['success'],1)]
    for kind,predicate,count in selections:
        candidates=[r for r in rows if r['episode_id'] not in used and predicate(r)]
        for r in rng.sample(candidates,min(count,len(candidates))):
            used.add(r['episode_id']);stem=f'{len(selected)+1:02d}_{kind}_{r["episode_id"]}';shutil.copy2(root/'episodes'/r['episode_id']/'rgb.mp4',inspection/(stem+'.mp4'))
            selected.append({'preview':stem+'.mp4','kind':kind,**{k:r[k] for k in ['episode_id','initial_state_id','split','category','success','failure_type','actions','frames','perturbation','attempt_results','events']}})
    (inspection/'selection.json').write_text(json.dumps(selected,indent=2))
    thumbrows=[]
    for r in selected:
        with h5py.File(root/'episodes'/r['episode_id']/'trajectory.h5','r') as f:
            im=f['observations/rgb'];indices=np.linspace(0,len(im)-1,5,dtype=int);canvas=Image.new('RGB',(1600,264),'#18202b');draw=ImageDraw.Draw(canvas)
            draw.text((8,6),f"{r['kind']} | {r['episode_id']} | success={r['success']} | {r['failure_type']}",fill='white')
            for j,idx in enumerate(indices):canvas.paste(Image.fromarray(im[idx]).resize((320,240)),(j*320,24))
            thumbrows.append(canvas)
    board=Image.new('RGB',(1600,len(thumbrows)*264))
    for i,row in enumerate(thumbrows):board.paste(row,(0,i*264))
    board.save(inspection/'contact_sheet.jpg',quality=90)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4));cats=list(bycat)
    axes[0].bar(cats,[statistics['category_outcomes'][c]['successes']/len(bycat[c]) for c in cats]);axes[0].set_ylim(0,1);axes[0].set_ylabel('Measured final success rate')
    failures=statistics['failure_types'];axes[1].bar(range(len(failures)),list(failures.values()));axes[1].set_xticks(range(len(failures)),list(failures),rotation=35,ha='right');axes[1].set_ylabel('Episodes')
    axes[2].hist(lengths,bins=25);axes[2].set_xlabel('Action steps (20 Hz)');fig.tight_layout();fig.savefig(inspection/'statistics.png',dpi=150);plt.close(fig)
    cards=''.join(f'<article><h2>{r["kind"]}: {r["episode_id"]}</h2><p>actual success={r["success"]}; failure={r["failure_type"]}; {r["actions"]} actions / {r["frames"]} RGB frames</p><video controls preload="metadata" src="{r["preview"]}"></video></article>' for r in selected)
    heading=f"RoboCasa: {len(rows)} trajectories, {statistics['successes']} success / {statistics['failures']} failure"
    selection_note='<p>Dataset composition is selected using measured final outcomes. Full source pool and selection rule are preserved in the dataset reports.</p>' if (root/'selection_manifest.json').exists() else ''
    (inspection/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+heading+'</title><style>body{font-family:system-ui;max-width:1200px;margin:30px auto;background:#f3f4f6}main{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{padding:16px;background:white;border-radius:10px}h2{font-size:18px}video{width:100%}</style><h1>'+heading+'</h1><p>These are unedited original simulation videos selected using actual outcomes. 20 FPS; T actions / T+1 RGB/state samples.</p>'+selection_note+'<main>'+cards+'</main>')
    print(json.dumps(statistics,indent=2))
if __name__=='__main__':main()
