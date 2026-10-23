#!/usr/bin/env bash
# Continue the verified one-update smoke checkpoint with optimizer state.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
id=${1:?complete experiment ID required}
[[ "$id" =~ ^nav-robothor-embclip-resume-smoke-sjc-[a-z0-9-]+$ ]] || exit 2
cd "$root"
out="$root/results/$id"
[[ ! -e "$out" ]] || { echo "Result leaf already exists: $out" >&2; exit 2; }
checkpoint="$root/results/nav-robothor-embclip-train-smoke-sjc-v1-seed0-20261019/allenact/checkpoints/ObjectNav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO/2026-10-19_23-53-24/exp_ObjectNav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000000007680.pt"
[[ -f "$checkpoint" ]] || exit 2
[[ "$(sha256sum "$checkpoint" | cut -d' ' -f1)" == d06367420b5d84af85bb0d7c967d8fe600177145d27409d34471983a6b132a3b ]] || exit 2
home="$root/assets/embclip-robothor/home"
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
  vendor/embclip-allenact/allenact/main.py scripts/embclip/sjc_resume_smoke_config.py -b . \
  --seed 0 --config_kwargs '{"headless":true,"train_gpu_ids":[0,1],"num_train_processes":60}' \
  -c "$checkpoint" -o "$out/allenact" \
  > "$out/train.log" 2>&1
rc=$?
set -e
printf '{"return_code": %s}\n' "$rc" > "$out/exit.json"
exit "$rc"
