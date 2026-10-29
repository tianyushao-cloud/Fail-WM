import json, time
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT = Path(__file__).resolve().parent
MODEL = Path('/media/tyshao/Jev-WM/paper_assets/decision_models/d3-turbo')
REVISION = 'a0f3e9e59ebd1202fe1d4de1c4c0f9a986758cbb'
started = time.time()
print(json.dumps({'stage':'download','revision':REVISION,'path':str(MODEL)}), flush=True)
p = snapshot_download(repo_id='vllm-sr/d3-turbo', revision=REVISION, local_dir=MODEL, max_workers=2, ignore_patterns=['assets/*', '.eval_results/*', '.gitattributes'])
(ROOT/'download.json').write_text(json.dumps({'repo':'vllm-sr/d3-turbo','revision':REVISION,'path':str(p),'seconds':time.time()-started},indent=2))
print('DOWNLOAD COMPLETE', flush=True)
