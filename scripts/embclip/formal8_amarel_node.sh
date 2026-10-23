#!/usr/bin/env bash
set -euo pipefail
: "${EMBCLIP_ROOT:?}"
: "${EMBCLIP_EXPERIMENT_ID:?}"
: "${EMBCLIP_MODEL:?}"
: "${EMBCLIP_SEED:?}"
: "${EMBCLIP_MASTER_HOST:?}"
: "${EMBCLIP_MASTER_PORT:?}"
: "${SLURM_PROCID:?}"
root=$EMBCLIP_ROOT
cd "$root"
out="$root/results/$EMBCLIP_EXPERIMENT_ID"
rank=$SLURM_PROCID
[[ "$rank" == 0 || "$rank" == 1 ]] || exit 2
export HOME="$root/assets/embclip-robothor/home"
export PATH="$root/assets/embclip-robothor/vulkan-tools/package/usr/bin:$PATH"
export PYTHONPATH="$root/vendor/embclip-allenact:$root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset DISPLAY
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv \
  > "$out/node-${rank}-gpus.csv"
config="scripts/embclip/formal8_${EMBCLIP_MODEL}_config.py"
set +e
"$root/.envs/embclip-robothor/bin/python" -u \
  vendor/embclip-allenact/allenact/main.py "$config" -b . \
  --seed "$EMBCLIP_SEED" \
  --machine_id "$rank" \
  --distributed_ip_and_port "${EMBCLIP_MASTER_HOST}:${EMBCLIP_MASTER_PORT}" \
  --config_kwargs '{"headless":true,"train_gpu_ids":[0,1,2,3],"num_train_processes":60}' \
  -o "$out/allenact" > "$out/node-${rank}.log" 2>&1
rc=$?
set -e
if grep -Eq 'Encountered (OutOfMemoryError|RuntimeError)|Uncaught exception|Traceback \(most recent call last\)' "$out/node-${rank}.log"; then
  rc=1
fi
exit "$rc"
