#!/usr/bin/env bash
# Run one complete official-policy PPO smoke with a clean checkpoint.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
id=${1:?complete experiment ID required}
[[ "$id" =~ ^nav-robothor-embclip-train-smoke-sjc-[a-z0-9-]+$ ]] || exit 2
cd "$root"
out="$root/results/$id"
[[ ! -e "$out" ]] || { echo "Result leaf already exists: $out" >&2; exit 2; }
[[ -f data/embclip-robothor/robothor-objectnav/train/episodes/FloorPlan_Train1_1.json.gz ]] || exit 2
home="$root/assets/embclip-robothor/home"
[[ -f "$home/.cache/clip/RN50.pt" ]] || exit 2
[[ -d "$home/.ai2thor/releases" ]] || exit 2
mkdir -p "$out"
export HOME="$home"
export PATH="$root/assets/embclip-robothor/vulkan-tools/package/usr/bin:$PATH"
export PYTHONPATH="$root/vendor/embclip-allenact:$root"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
unset DISPLAY
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv > "$out/gpu-before.csv"
set +e
timeout --signal=TERM --kill-after=30 3540 "$root/.envs/embclip-robothor/bin/python" -u \
  vendor/embclip-allenact/allenact/main.py scripts/embclip/sjc_train_smoke_config.py -b . \
  --seed 0 --config_kwargs '{"headless":true,"train_gpu_ids":[0,1],"num_train_processes":60}' \
  -o "$out/allenact" \
  > "$out/train.log" 2>&1
rc=$?
set -e
printf '{"return_code": %s}\n' "$rc" > "$out/exit.json"
exit "$rc"
