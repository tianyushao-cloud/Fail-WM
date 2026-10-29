import argparse,json
from pathlib import Path
from rc_collect.environment import make_env
from rc_collect.controller import ScriptedController
from rc_collect.recording import Recorder
from rc_collect.perturbations import BRANCHES
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='data/preview5');ap.add_argument('--only',default='');args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);summ=[]
    for i,(kind,pert) in enumerate(BRANCHES,1):
        if args.only and args.only!=kind:continue
        e=make_env();rec=Recorder(e,out/f'{i:02d}_{kind}',kind);ScriptedController(e,rec.step).run(kind);m=rec.finish(pert);e.close();summ.append(m)
        print('RESULT',json.dumps({k:m[k] for k in ('episode_id','success','failure_type','actions','frames','event_summary')}),flush=True)
    (out/'summary.json').write_text(json.dumps(summ,indent=2))
if __name__=='__main__':main()
