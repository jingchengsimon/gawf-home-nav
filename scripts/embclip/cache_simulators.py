"""Download simulator builds on the login node without starting Unity."""
import hashlib
import json
from pathlib import Path
import traceback

import ai2thor.build
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


root = Path(__file__).resolve().parents[2]
cache = root / 'assets/embclip-robothor/ai2thor'
cache.mkdir(parents=True, exist_ok=True)
Controller.base_dir = property(lambda self: str(cache))
rows = []
for platform, commit, headless in [
    (CloudRendering, ai2thor.build.COMMIT_ID, False),
    (CloudRendering, '91139c909576f3bf95a187c5b02c6fd455d06b48', False),
]:
    row = {'platform': platform.__name__, 'commit': commit, 'download_only': True}
    try:
        controller = Controller(platform=platform, commit_id=commit,
                                headless=headless, download_only=True,
                                include_private_scenes=False)
        build = controller._build
        row.update(status='cached', executable=build.executable_path)
        executable = Path(build.executable_path)
        assert executable.is_file()
        row['executable_sha256'] = hashlib.sha256(executable.read_bytes()).hexdigest()
    except Exception:
        row.update(status='unavailable', error=traceback.format_exc())
    rows.append(row)
    print(json.dumps(row), flush=True)
receipt = root / 'logs/embclip-robothor-setup/simulator-cache-v2.json'
receipt.write_text(json.dumps(rows, indent=2) + '\n')
assert rows[0]['status'] == 'cached', 'CloudRendering build must be cached before GPU preflight'
