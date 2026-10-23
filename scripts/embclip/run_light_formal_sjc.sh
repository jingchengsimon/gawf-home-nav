#!/usr/bin/env bash
# Single-GPU 1M-step light EmbCLIP training on SJC; safe to share a physical GPU.
set -euo pipefail
root=${EMBCLIP_ROOT:?Set EMBCLIP_ROOT to the SJC project root}
model=${1:?gru1, gru3, or lstm3 required}
id=${2:?complete experiment ID required}
gpu=${3:?physical GPU index required}
[[ "$model" == gru1 || "$model" == gru3 || "$model" == lstm3 ]] || exit 2
[[ "$gpu" == 0 || "$gpu" == 1 ]] || exit 2
[[ "$id" =~ ^nav-robothor-embclip-light-${model}-formal1m-sjc-v1-seed0-[0-9]{8}$ ]] || exit 2
cd "$root"
test -d "vendor/embclip-allenact"
test -f "data/embclip-robothor/robothor-objectnav/train/episodes/FloorPlan_Train1_1.json.gz"
test -f "assets/embclip-robothor/home/.cache/clip/RN50.pt"
test -x ".envs/embclip-robothor/bin/python"
out="$root/results/$id"
test -d "$out"
test ! -e "$out/exit.json"
"$root/.envs/embclip-robothor/bin/python" - "$id" "$model" "$gpu" <<'PY'
import json, sys
from pathlib import Path
run = json.loads((Path('experiments/monitoring/jobs') / (sys.argv[1] + '.json')).read_text())
assert run['experiment_id'] == sys.argv[1] and run['host'] == 'sjc-remote'
assert run['model']['family'] == sys.argv[2]
assert run['execution']['physical_gpu_index'] == int(sys.argv[3])
assert run['execution']['steps_budget'] == 1_000_000
assert run['execution']['global_train_envs'] == 16
PY
export HOME="$root/assets/embclip-robothor/home"
export PATH="$root/assets/embclip-robothor/vulkan-tools/package/usr/bin:$PATH"
export PYTHONPATH="$root/vendor/embclip-allenact:$root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset DISPLAY
printf 'started_at=%s\nphysical_gpu=%s\n' "$(date -Is)" "$gpu" > "$out/allocation.txt"
finish() {
  rc=$?
  printf '{"return_code": %s}\n' "$rc" > "$out/exit.json"
}
trap finish EXIT
"$root/.envs/embclip-robothor/bin/python" -u \
  vendor/embclip-allenact/allenact/main.py "scripts/embclip/light_${model}_config.py" -b . \
  --seed 0 --config_kwargs '{"headless":true,"train_gpu_ids":[0],"num_train_processes":16}' \
  -o "$out/allenact" > "$out/train.log" 2>&1
"$root/.envs/embclip-robothor/bin/python" - "$out/allenact/checkpoints" <<'PY'
from pathlib import Path
import re, sys
steps = [int(m.group(1)) for p in Path(sys.argv[1]).rglob('*.pt')
         if (m := re.search(r'__steps_(\d+)\.pt$', p.name))]
assert steps and max(steps) >= 1_000_000, f'No final 1M checkpoint; last={max(steps) if steps else None}'
print(f'final_checkpoint_steps={max(steps)}')
PY
