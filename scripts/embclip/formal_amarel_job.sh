#!/usr/bin/env bash
# Four-node, sixteen-GPU formal EmbCLIP run. No simulator smoke is launched here.
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the project root}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
root=$SLURM_SUBMIT_DIR
id=${1:?complete experiment ID required}
model=${2:?gru1, gru3, or gawf3 required}
seed=${3:?seed required}
[[ "$model" == gru1 || "$model" == gru3 || "$model" == gawf3 ]] || exit 2
[[ "$seed" == 0 || "$seed" == 1 || "$seed" == 2 ]] || exit 2
[[ "$id" =~ ^nav-robothor-embclip-${model}-formal-200m-v1-seed${seed}-[0-9]{8}$ ]] || exit 2
[[ "${SLURM_JOB_NUM_NODES:-}" == 4 ]] || { echo 'Expected four nodes' >&2; exit 2; }
cd "$root"
test "$(git -C vendor/embclip-allenact rev-parse HEAD)" = 855fe741ddecaaf2492966d52d3f130e58d3c48f
test -z "$(git -C vendor/embclip-allenact status --porcelain=v1 --untracked-files=no)"
sha256sum -c configs/embclip/formal-source.sha256
test -f data/embclip-robothor/robothor-objectnav/train/episodes/FloorPlan_Train1_1.json.gz
test -f assets/embclip-robothor/home/.cache/clip/RN50.pt
test -d assets/embclip-robothor/home/.ai2thor/releases/thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917
test -x assets/embclip-robothor/vulkan-tools/package/usr/bin/vulkaninfo
test -x .envs/embclip-robothor/bin/python
test -d "results/$id"
out="$root/results/$id"
export EMBCLIP_ROOT="$root"
export EMBCLIP_EXPERIMENT_ID="$id"
export EMBCLIP_MODEL="$model"
export EMBCLIP_SEED="$seed"
export EMBCLIP_MASTER_HOST
EMBCLIP_MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export EMBCLIP_MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))
printf 'job_id=%s\nmaster=%s\nport=%s\nnodes=%s\n' \
  "$SLURM_JOB_ID" "$EMBCLIP_MASTER_HOST" "$EMBCLIP_MASTER_PORT" \
  "$SLURM_JOB_NODELIST" > "$out/allocation.txt"
trap 'rc=$?; printf "{\"return_code\": %s}\n" "$rc" > "$out/exit.json"' EXIT
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpu-bind=none \
  bash scripts/embclip/formal_amarel_node.sh
"$root/.envs/embclip-robothor/bin/python" - "$out/allenact/checkpoints" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
steps = []
for path in root.rglob('*.pt'):
    match = re.search(r'__steps_(\d+)\.pt$', path.name)
    if match:
        steps.append(int(match.group(1)))
assert steps and max(steps) >= 200_000_000, f'No final 200M checkpoint; last={max(steps) if steps else None}'
print(f'final_checkpoint_steps={max(steps)}')
PY
