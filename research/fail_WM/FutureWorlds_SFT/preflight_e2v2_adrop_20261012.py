"""CPU-only audit of the exact Opus 2026-10-12 sampling proposal."""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from futureworlds.sampling import Schedule

ROOT = Path('/media/tyshao/fail_WM/FutureWorlds_SFT')
OUT = ROOT / 'experiments_e2v2_adrop_20261012'
EVENTS = {'contact', 'grasp', 'lift', 'drop', 'release', 'grasp_attempt_failed'}

def save(name, obj):
    p = OUT / name
    text = json.dumps(obj, ensure_ascii=False, indent=2) + '\n'
    if p.exists() and p.read_text() != text:
        raise ValueError(f'Refusing to replace different result: {p}')
    p.write_text(text)

def main():
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((ROOT/'experiments_mixed_codec_wm_20261011/natural.json').read_text())
    cases = json.loads(Path(cfg['train_manifest']).read_text())['cases']
    source = Path('/media/tyshao/fail_WM/RoboCasa/data/batch1000/manifest.jsonl')
    raw = {r['episode_id']: r for r in map(json.loads, source.read_text().splitlines())}
    meta = {}
    for row in cases:
        s = raw[row['id']]
        phases = s.get('phase_ranges', {})
        lift = phases.get('attempt0/lift')
        release = phases.get('attempt0/early_release')
        L = lift['first_action']//4+1 if lift else None
        anchors = [(math.ceil(e['state_index']/4), 4) for e in s['events']
                   if e['event'] in EVENTS and 'state_index' in e]
        if lift: anchors.append((L, 6))
        if release: anchors.append((release['first_action']//4+1, 6))
        window = sorted({t for a, after in anchors for t in range(a-2,a+after+1)
                         if 2 <= t < row['frames']})
        grip = 'missing_phase'
        if lift:
            with np.load(Path(cfg['train_manifest']).parent/row['file']) as z:
                actions = z['actions'].reshape(-1,13)
                vals = actions[lift['first_action']:lift['last_action']+1,6]
            grip = '+1' if np.allclose(vals,1) else '0' if np.allclose(vals,0) else 'other'
        never = not bool(s['event_summary']['lift'])
        groups = ['all', 'success' if row['success'] else 'failure',
                  'gripper/'+grip, 'outcome/'+('never_lift' if never else 'lifted')+'/gripper/'+grip]
        if not row['success']: groups.append('failure_type/'+str(row['failure_type']))
        critical = not row['success'] and never and grip == '+1'
        if critical: groups.append('critical_closed_never_lift_failure')
        meta[row['id']] = dict(L=L,window=window,gripper=grip,never_lift=never,
                              success=row['success'],groups=groups,frames=row['frames'])
    save('episode_windows.json',meta)
    print('episode strata',dict(Counter((m['gripper'],m['never_lift']) for m in meta.values())),flush=True)
    stats = defaultdict(Counter)
    for m in meta.values():
        for g in m['groups']:
            stats[g]['episodes'] += 1
            stats[g]['missing_lift_phase'] += m['L'] is None
    sampler = Schedule(cases,'robocasa',cfg['seed'])
    schedule = []
    for cursor in range(cfg['steps']*cfg['global_batch']):
        row, original = sampler.pick(cursor)
        m = meta[row['id']]
        rng = random.Random(cfg['seed'] + cursor*104729 + 20261012)
        forced = bool(m['window']) and rng.random()<0.5
        target = m['window'][rng.randrange(len(m['window']))] if forced else original
        drop = random.Random(cfg['seed'] + cursor*130363 + 60261012).random()<0.25
        L = m['L']
        in_base = L is not None and L <= original <= L+6
        in_new = L is not None and L <= target <= L+6
        n = sum(L is not None and L <= t <= L+6 for t in m['window'])
        q = n/len(m['window']) if m['window'] else 0
        for g in m['groups']:
            c=stats[g];c['samples']+=1;c['baseline_in_L_L6']+=in_base
            c['e2v2_in_L_L6']+=in_new;c['forced']+=forced;c['dropout']+=drop
            c['expected_pure_window_hits']+=q
        assert 2 <= target < row['frames']
        schedule.append({'id':row['id'],'target':target})
        if cursor and cursor%160000==0: print('cursors',cursor,flush=True)
    for c in stats.values():
        c['baseline_fraction']=c['baseline_in_L_L6']/c['samples']
        c['e2v2_fraction']=c['e2v2_in_L_L6']/c['samples']
        c['exposure_ratio']=c['e2v2_in_L_L6']/c['baseline_in_L_L6'] if c['baseline_in_L_L6'] else None
        c['dropout_fraction']=c['dropout']/c['samples']
    cache_checks = Counter()
    ranges = {}
    for p in sorted(Path(cfg['token_cache']).glob('*.npz')):
        with np.load(p) as z:
            for key in ('context','dynamics','actions'):
                a = z[key]
                if key == 'context': a=a+4375
                cache_checks['null9006_occurrences'] += int(np.count_nonzero(a==9006))
                lo,hi = int(a.min()),int(a.max())
                old=ranges.get(key,[lo,hi]);ranges[key]=[min(old[0],lo),max(old[1],hi)]
            cache_checks['episodes']+=1
    critical=stats['critical_closed_never_lift_failure']
    gate=critical['episodes']==202 and critical['exposure_ratio']>=3
    save('schedule_e2v2_proposed.json',schedule)
    result=dict(status='PASS' if gate else 'HOLD_COVERAGE_GATE',stats=dict(stats),
                cache_scan=dict(cache_checks),cache_ranges=ranges,
                schedule_entries=len(schedule),
                schedule_sha256=hashlib.sha256((OUT/'schedule_e2v2_proposed.json').read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                schedule_rng='random.Random(seed + cursor*104729 + 20261012)',
                dropout_rng='random.Random(seed + cursor*130363 + 60261012)',
                critical_definition='train failure; no event_summary.lift in entire episode; all raw-index aligned lift phase gripper controls equal +1',
                training_started=False)
    save('preflight.json',result)
    print(json.dumps({k:result[k] for k in ('status','cache_scan','cache_ranges')},indent=2))
    print('CRITICAL',json.dumps(critical,indent=2))

if __name__=='__main__': main()
