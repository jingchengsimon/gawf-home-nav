#!/usr/bin/env bash
# Single-node, single-GPU 1M-step EmbCLIP comparison; no smoke on Amarel.
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the project root}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
root=$SLURM_SUBMIT_DIR
id=${1:?complete experiment ID required}
model=${2:?gru1, gru3, or lstm3 required}
seed=${3:?seed required}
[[ "$model" == gru1 || "$model" == gru3 || "$model" == lstm3 ]] || exit 2
[[ "$seed" == 0 ]] || exit 2
[[ "$id" =~ ^nav-robothor-embclip-light-${model}-formal1m-1gpu-v1-seed0-[0-9]{8}$ ]] || exit 2
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 ]] || exit 2
cd "$root"
test "$(git -C vendor/embclip-allenact rev-parse HEAD)" = 855fe741ddecaaf2492966d52d3f130e58d3c48f
test -z "$(git -C vendor/embclip-allenact status --porcelain=v1 --untracked-files=no)"
sha256sum -c configs/embclip/light-source.sha256
test -f data/embclip-robothor/robothor-objectnav/train/episodes/FloorPlan_Train1_1.json.gz
test -f assets/embclip-robothor/home/.cache/clip/RN50.pt
test -d assets/embclip-robothor/home/.ai2thor/releases/thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917
test -x assets/embclip-robothor/vulkan-tools/package/usr/bin/vulkaninfo
test -x .envs/embclip-robothor/bin/python
out="$root/results/$id"
test -d "$out"
test ! -e "$out/exit.json"
"$root/.envs/embclip-robothor/bin/python" - "$id" "$model" "$SLURM_JOB_ID" <<'PY'
import json, sys
from pathlib import Path
root = Path.cwd()
run = json.loads((root / 'experiments/monitoring/jobs' / (sys.argv[1] + '.json')).read_text())
assert run['experiment_id'] == sys.argv[1] and run['host'] == 'amarel'
assert run['model']['family'] == sys.argv[2] and run['execution']['job_id'] == sys.argv[3]
assert run['execution']['steps_budget'] == 1_000_000
assert run['execution']['global_train_envs'] == 16
PY
export HOME="$root/assets/embclip-robothor/home"
export PATH="$root/assets/embclip-robothor/vulkan-tools/package/usr/bin:$PATH"
export PYTHONPATH="$root/vendor/embclip-allenact:$root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset DISPLAY
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$out/gpu-before.csv"
printf 'job_id=%s\nstarted_at=%s\n' "$SLURM_JOB_ID" "$(date -Is)" > "$out/allocation.txt"
finish() {
  rc=$?
  printf '{"return_code": %s}\n' "$rc" > "$out/exit.json"
}
trap finish EXIT
"$root/.envs/embclip-robothor/bin/python" -u \
  vendor/embclip-allenact/allenact/main.py "scripts/embclip/light_${model}_config.py" -b . \
  --seed "$seed" \
  --config_kwargs '{"headless":true,"train_gpu_ids":[0],"num_train_processes":16}' \
  -o "$out/allenact" > "$out/train.log" 2>&1
"$root/.envs/embclip-robothor/bin/python" - "$out/allenact/checkpoints" <<'PY'
from pathlib import Path
import re, sys
steps = []
for path in Path(sys.argv[1]).rglob('*.pt'):
    match = re.search(r'__steps_(\d+)\.pt$', path.name)
    if match:
        steps.append(int(match.group(1)))
assert steps and max(steps) >= 1_000_000, f'No final 1M checkpoint; last={max(steps) if steps else None}'
print(f'final_checkpoint_steps={max(steps)}')
PY
