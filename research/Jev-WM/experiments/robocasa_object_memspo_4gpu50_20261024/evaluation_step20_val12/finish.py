import json,time,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
while True:
 status=json.loads((R/'status.json').read_text())
 if status['state']=='FAILED':raise RuntimeError(status)
 if status['state']=='COMPLETE':
  subprocess.run([sys.executable,str(R/'report.py')],check=True)
  (R/'report_complete.json').write_text(json.dumps(dict(state='COMPLETE',paired_cases=12,videos=12)))
  break
 time.sleep(15)
