#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
id=${1:?complete experiment ID required}
[[ "$id" =~ ^nav-robothor-embclip-render-[a-z0-9-]+$ ]] || exit 2
cd "$root"
out="$root/results/$id"
mkdir -p "$out"
unset DISPLAY
ldconfig -p 2>/dev/null | grep -E 'vulkan|libGLX_nvidia|libnvidia' > "$out/libraries.txt" || true
for folder in /etc/vulkan/icd.d /usr/share/vulkan/icd.d; do
  if [[ -d "$folder" ]]; then ls -l "$folder"; fi
done > "$out/vulkan-icds.txt" 2>&1
curl -I --max-time 15 https://ai2thor-builds.s3-us-west-2.amazonaws.com/ > "$out/compute-network.txt" 2>&1 || true
export PYTHONUNBUFFERED=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
set +e
timeout --signal=TERM --kill-after=15 240 "$root/.envs/embclip-robothor/bin/python" -u scripts/embclip/render_check.py --output "$out/cloud" --mode cloud
rc=$?
if [[ "$rc" != 0 ]]; then
  timeout --signal=TERM --kill-after=15 240 "$root/.envs/embclip-robothor/bin/python" -u scripts/embclip/render_check.py --output "$out/legacy-headless" --mode legacy-headless
  rc=$?
fi
set -e
"$root/.envs/embclip-robothor/bin/python" - "$out" "$rc" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rc=int(sys.argv[2]); attempts=[]
for mode in ['cloud','legacy-headless']:
    path=root/mode/'summary.json'
    if path.is_file(): attempts.append(json.loads(path.read_text()))
summary={'status':'complete' if rc==0 else 'failed','completed_checks':int(rc==0),'attempts':attempts}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'exit.json').write_text(json.dumps({'return_code':rc,'error':None if rc==0 else 'Both permitted rendering paths failed; see attempt error.txt and Slurm logs'})+'\n')
PY
exit "$rc"
