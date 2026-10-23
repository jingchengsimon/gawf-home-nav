#!/usr/bin/env bash
# Foreground RoboTHOR rendering check on the SJC GPU workstation.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
id=${1:?complete experiment ID required}
gpu=${2:?GPU index required}
[[ "$id" =~ ^nav-robothor-embclip-render-sjc-[a-z0-9-]+$ ]] || exit 2
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
cd "$root"
out="$root/results/$id"
[[ ! -e "$out" ]] || { echo "Result leaf already exists: $out" >&2; exit 2; }
[[ -x "$root/.envs/embclip-robothor/bin/python" ]] || { echo 'EmbCLIP runtime is missing' >&2; exit 2; }
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES="$gpu"
unset DISPLAY
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv > "$out/gpu-before.csv"
ldconfig -p 2>/dev/null | grep -E 'vulkan|libGLX_nvidia|libnvidia' > "$out/libraries.txt" || true
for folder in /etc/vulkan/icd.d /usr/share/vulkan/icd.d; do
  if [[ -d "$folder" ]]; then ls -l "$folder"; fi
done > "$out/vulkan-icds.txt" 2>&1
export PYTHONUNBUFFERED=1
set +e
timeout --signal=TERM --kill-after=15 240 "$root/.envs/embclip-robothor/bin/python" -u \
  scripts/embclip/render_check_sjc.py --output "$out/cloud" --mode cloud \
  > "$out/cloud.log" 2>&1
rc=$?
if [[ "$rc" != 0 ]]; then
  timeout --signal=TERM --kill-after=15 240 "$root/.envs/embclip-robothor/bin/python" -u \
    scripts/embclip/render_check_sjc.py --output "$out/legacy-headless" --mode legacy-headless \
    > "$out/legacy-headless.log" 2>&1
  rc=$?
fi
set -e
"$root/.envs/embclip-robothor/bin/python" - "$out" "$rc" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); rc = int(sys.argv[2]); attempts = []
for mode in ('cloud', 'legacy-headless'):
    path = root / mode / 'summary.json'
    if path.is_file(): attempts.append(json.loads(path.read_text()))
summary = {'status': 'complete' if rc == 0 else 'failed',
           'completed_checks': int(rc == 0), 'attempts': attempts}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(root / 'exit.json').write_text(json.dumps({'return_code': rc}) + '\n')
PY
exit "$rc"
