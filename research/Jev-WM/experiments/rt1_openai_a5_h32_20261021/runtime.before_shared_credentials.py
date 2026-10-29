"""Load existing private credential only into environment; never serialize it."""
import os
from pathlib import Path


def api_environment():
    if not os.environ.get('OPENAI_API_KEY'):
        key_file = Path(os.environ.get('JEV_API_KEY_FILE','/root/.config/jev-wm/openai.key'))
        if not key_file.is_file():
            raise RuntimeError('Set OPENAI_API_KEY or JEV_API_KEY_FILE to the private credential file')
        os.environ['OPENAI_API_KEY'] = key_file.read_text().strip()
    proxy = os.environ.get('JEV_HTTPS_PROXY','http://127.0.0.1:17897')
    for name in ['https_proxy','http_proxy','HTTPS_PROXY','HTTP_PROXY']:
        if proxy:
            os.environ[name] = proxy
        else:
            os.environ.pop(name,None)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost'
