import time,json,subprocess,sys
from pathlib import Path
R=Path(__file__).parent
while True:
 status=json.loads((R/'status.json').read_text())
 if status['state']=='COMPLETE':
  subprocess.run([sys.executable,str(R/'summarize.py')],check=True)
  break
 if status['state']=='FAILED':
  raise RuntimeError(status)
 time.sleep(20)
