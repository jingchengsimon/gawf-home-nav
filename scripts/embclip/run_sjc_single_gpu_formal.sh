#!/usr/bin/env bash
set -euo pipefail

root=${EMBCLIP_ROOT:?Set EMBCLIP_ROOT to the SJC project root}
adapter=${EMBCLIP_ADAPTER:?Set EMBCLIP_ADAPTER to the experiment adapter root}
model=${1:?gru3 or gawf3 required}
id=${2:?complete experiment ID required}
gpu=${3:?physical GPU index required}
[[ "$model" == gru3 || "$model" == gawf3 ]] || exit 2
[[ "$gpu" == 0 || "$gpu" == 1 ]] || exit 2
[[ "$id" =~ ^nav-robothor-embclip-${model}-formal-200m-sjc-v1-seed0-[0-9]{8}$ ]] || exit 2

cd "$root"
out="$root/results/$id"
[[ -d "$out" && -f "$out/prepared.json" && ! -e "$out/exit.json" ]] || exit 2
test -f data/embclip-robothor/robothor-objectnav/train/episodes/FloorPlan_Train1_1.json.gz
test -f assets/embclip-robothor/home/.cache/clip/RN50.pt
test -d assets/embclip-robothor/home/.ai2thor/releases/thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917
test -x assets/embclip-robothor/vulkan-tools/package/usr/bin/vulkaninfo
test -x .envs/embclip-robothor/bin/python
test -f "$adapter/scripts/embclip/sjc_${model}_formal_config.py"

export HOME="$root/assets/embclip-robothor/home"
export PATH="$root/assets/embclip-robothor/vulkan-tools/package/usr/bin:$PATH"
export PYTHONPATH="$adapter:$root/vendor/embclip-allenact:$root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset DISPLAY
unset CUDA_VISIBLE_DEVICES

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$out/gpu-before.csv"
printf '%s\n' "$(date -Is)" > "$out/started-at.txt"
finish() {
  rc=$?
  printf '{"return_code": %s}\n' "$rc" > "$out/exit.json"
}
trap finish EXIT
"$root/.envs/embclip-robothor/bin/python" -u \
  vendor/embclip-allenact/allenact/main.py "scripts/embclip/sjc_${model}_formal_config.py" -b . \
  --seed 0 \
  --config_kwargs "{\"headless\":true,\"train_gpu_ids\":[${gpu}],\"num_train_processes\":60}" \
  -o "$out/allenact" > "$out/train.log" 2>&1
