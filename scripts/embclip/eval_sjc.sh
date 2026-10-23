#!/usr/bin/env bash
# Evaluate the frozen official checkpoint on a recorded validation subset.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
id=${1:?complete experiment ID required}
[[ "$id" =~ ^nav-robothor-embclip-eval-sjc-[a-z0-9-]+$ ]] || exit 2
cd "$root"
out="$root/results/$id"
[[ ! -e "$out" ]] || { echo "Result leaf already exists: $out" >&2; exit 2; }
[[ -f data/embclip-robothor/eval-smoke-val100-v1/receipt.json ]] || exit 2
[[ -f assets/embclip-robothor/checkpoints/clip-rn50-gru-130091717.pt ]] || exit 2
home="$root/assets/embclip-robothor/home"
mkdir -p "$home/.cache/clip"
if [[ ! -e "$home/.cache/clip/RN50.pt" ]]; then
  ln -s "$root/assets/embclip-robothor/clip/RN50.pt" "$home/.cache/clip/RN50.pt"
fi
if [[ ! -e "$home/.ai2thor" ]]; then
  ln -s "$root/assets/embclip-robothor/ai2thor" "$home/.ai2thor"
fi
mkdir -p "$out"
export HOME="$home"
export PATH="$root/assets/embclip-robothor/vulkan-tools/package/usr/bin:$PATH"
export PYTHONPATH="$root/vendor/embclip-allenact:$root"
export EMBCLIP_VAL_DATASET_DIR="$root/data/embclip-robothor/eval-smoke-val100-v1"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset DISPLAY
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv > "$out/gpu-before.csv"
set +e
timeout --signal=TERM --kill-after=30 3600 "$root/.envs/embclip-robothor/bin/python" -u \
  vendor/embclip-allenact/allenact/main.py scripts/embclip/sjc_config.py -b . \
  --eval --seed 0 \
  --config_kwargs '{"headless":true,"test_on_validation":true,"test_gpu_ids":[0],"num_test_processes":15}' \
  -c "$root/assets/embclip-robothor/checkpoints/clip-rn50-gru-130091717.pt" \
  -o "$out/allenact" \
  > "$out/eval.log" 2>&1
rc=$?
set -e
printf '{"return_code": %s}\n' "$rc" > "$out/exit.json"
exit "$rc"
