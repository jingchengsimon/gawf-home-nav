#!/usr/bin/env bash
# One PPO rollout for a new comparison model, isolated from the SJC checkout.
set -euo pipefail
root=${EMBCLIP_ROOT:?Set EMBCLIP_ROOT to the SJC project root}
adapter=${EMBCLIP_ADAPTER:?Set EMBCLIP_ADAPTER to the temporary adapter root}
model=${1:?gru3 or gawf3 required}
id=${2:?complete smoke experiment ID required}
[[ "$model" == gru3 || "$model" == gawf3 ]] || exit 2
[[ "$id" =~ ^nav-robothor-embclip-${model}-smoke-sjc-v[0-9]+-seed0-[0-9]{8}$ ]] || exit 2
cd "$root"
out="$root/results/$id"
[[ ! -e "$out" ]] || { echo "Existing result leaf: $out" >&2; exit 2; }
[[ -f data/embclip-robothor/robothor-objectnav/train/episodes/FloorPlan_Train1_1.json.gz ]] || exit 2
[[ -f assets/embclip-robothor/home/.cache/clip/RN50.pt ]] || exit 2
[[ -d assets/embclip-robothor/home/.ai2thor/releases ]] || exit 2
[[ -x "$root/.envs/embclip-robothor/bin/python" ]] || exit 2
[[ -f "$adapter/scripts/embclip/sjc_${model}_smoke_config.py" ]] || exit 2
mkdir -p "$out"
export HOME="$root/assets/embclip-robothor/home"
export PATH="$root/assets/embclip-robothor/vulkan-tools/package/usr/bin:$PATH"
export PYTHONPATH="$adapter:$root/vendor/embclip-allenact:$root"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
unset DISPLAY
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv > "$out/gpu-before.csv"
set +e
timeout --signal=TERM --kill-after=30 3540 "$root/.envs/embclip-robothor/bin/python" -u \
  vendor/embclip-allenact/allenact/main.py "scripts/embclip/sjc_${model}_smoke_config.py" -b . \
  --seed 0 --config_kwargs '{"headless":true,"train_gpu_ids":[0,1],"num_train_processes":60}' \
  -o "$out/allenact" > "$out/train.log" 2>&1
rc=$?
set -e
printf '{"return_code": %s}\n' "$rc" > "$out/exit.json"
exit "$rc"
