"""Validate frame alignment and current-qpos geometry via native-action replay."""
import argparse,json
from pathlib import Path
from rc_collect.batch_validation import validate_episode
from rc_collect.environment import make_env

def validate(path,replay=True):
    m=json.loads((path/'metadata.json').read_text());e=None
    try:
        if replay:e=make_env(seed=m.get('seed',20261003),initial_spec=m.get('initial_spec',{}))
        return {'episode':path.name,**validate_episode(path,e)}
    finally:
        if e is not None:e.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/preview5');p.add_argument('--no-replay',action='store_true');a=p.parse_args();root=Path(a.data)
    folders=root/'episodes' if (root/'episodes').exists() else root
    reports=[validate(x,not a.no_replay) for x in sorted(folders.iterdir()) if (x/'trajectory.h5').exists()]
    (root/'verification.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
