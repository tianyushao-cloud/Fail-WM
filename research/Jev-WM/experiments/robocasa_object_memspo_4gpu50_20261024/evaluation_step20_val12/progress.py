import json
from pathlib import Path
import numpy as np
R=Path(__file__).parent;p=json.loads((R/'protocol.json').read_text());pairs=[]
for row in p['cases']:
 paths=[R/'results'/row['id']/n/'result.json' for n in ['sft','rl20']]
 if not all(x.exists() for x in paths):continue
 a,b=[json.loads(x.read_text()) for x in paths]
 entry=dict(id=row['id'],success=row['success'],sft_lpips=a['metrics']['32']['lpips'],rl20_lpips=b['metrics']['32']['lpips'],lpips_gain=a['metrics']['32']['lpips']-b['metrics']['32']['lpips'],roi_gain=a['roi_metrics']['lpips']-b['roi_metrics']['lpips'],track_error_reduction=a['object']['track_error_px_capped100']-b['object']['track_error_px_capped100'],missing_before=a['object']['missing_rate'],missing_after=b['object']['missing_rate'])
 pairs.append(entry)
print(json.dumps(dict(status=json.loads((R/'status.json').read_text()),paired_cases=len(pairs),pairs=pairs),indent=2))
