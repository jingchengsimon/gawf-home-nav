#!/usr/bin/env bash
set -euo pipefail
root=${EMBCLIP_ROOT:?Set EMBCLIP_ROOT to the SJC project root}
model=${1:?gru1, gru3, or lstm3 required}
id=${2:?complete smoke experiment ID required}
gpu=${3:?physical GPU index required}
[[ "$model" == gru1 || "$model" == gru3 || "$model" == lstm3 ]] || exit 2
[[ "$gpu" == 0 || "$gpu" == 1 ]] || exit 2
[[ "$id" =~ ^nav-robothor-embclip-light-${model}-smoke-sjc-v1-seed0-[0-9]{8}$ ]] || exit 2
cd "$root"
out="$root/results/$id"
[[ ! -e "$out" ]] || { echo "Existing result leaf: $out" >&2; exit 2; }
test -f data/embclip-robothor/robothor-objectnav/train/episodes/FloorPlan_Train1_1.json.gz
test -f assets/embclip-robothor/home/.cache/clip/RN50.pt
test -x .envs/embclip-robothor/bin/python
mkdir -p "$out"
export HOME="$root/assets/embclip-robothor/home"
export PATH="$root/assets/embclip-robothor/vulkan-tools/package/usr/bin:$PATH"
export PYTHONPATH="$root/vendor/embclip-allenact:$root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset DISPLAY
printf '%s\n' "$(date -Is)" > "$out/started-at.txt"
set +e
timeout --signal=TERM --kill-after=30 1200 \
  "$root/.envs/embclip-robothor/bin/python" -u \
  vendor/embclip-allenact/allenact/main.py "scripts/embclip/light_${model}_smoke_sjc_config.py" -b . \
  --seed 0 --config_kwargs '{"headless":true,"train_gpu_ids":[0],"num_train_processes":16}' \
  -o "$out/allenact" > "$out/train.log" 2>&1
rc=$?
set -e
printf '{"return_code": %s}\n' "$rc" > "$out/exit.json"
exit "$rc"
