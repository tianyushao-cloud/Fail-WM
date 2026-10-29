#!/usr/bin/env python3
"""Start/resume server-only evaluation independently of an SSH session."""
import json
import os
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evals_e1_expanded_20261009'
PYTHON='/media/tyshao/fail_WM/FutureWorlds_workspace/code/.venv/bin/python'
pid_file=OUT/'worker.pid'
if pid_file.exists():
    pid=int(pid_file.read_text());cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists() and b'expanded_e1.py' in cmd.read_bytes():
        print(json.dumps({'state':'already_running','pid':pid}));raise SystemExit(0)
env=dict(os.environ,PYTHONPATH='/media/tyshao/fail_WM/FutureWorlds_workspace/code/src:'+str(ROOT),USE_TF='0',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
with (OUT/'run.log').open('ab',buffering=0) as log:
    child=subprocess.Popen([PYTHON,'-u',str(ROOT/'expanded_e1.py'),'run'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pid_file.write_text(str(child.pid)+'\n')
print(json.dumps({'state':'started','pid':child.pid,'log':str(OUT/'run.log')}))
