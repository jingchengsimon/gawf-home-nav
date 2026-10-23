#!/usr/bin/env bash
# Continue the same smoke to ten global rollouts, with bounded telemetry.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
id=${1:?complete experiment ID required}
[[ "$id" =~ ^nav-robothor-embclip-profile-smoke-sjc-[a-z0-9-]+$ ]] || exit 2
cd "$root"
out="$root/results/$id"
[[ ! -e "$out" ]] || { echo "Result leaf already exists: $out" >&2; exit 2; }
checkpoint="$root/results/nav-robothor-embclip-resume-smoke-sjc-v1-seed0-20261019/allenact/checkpoints/ObjectNav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO/2026-10-19_23-57-12/exp_ObjectNav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000000015360.pt"
[[ -f "$checkpoint" ]] || exit 2
[[ "$(sha256sum "$checkpoint" | cut -d' ' -f1)" == a3b903c300f7d9c2e532a71ff70d1c2bd4cc9a4e2dbecf9514c473a636cf6e73 ]] || exit 2
mkdir -p "$out"
export HOME="$root/assets/embclip-robothor/home"
export PATH="$root/assets/embclip-robothor/vulkan-tools/package/usr/bin:$PATH"
export PYTHONPATH="$root/vendor/embclip-allenact:$root"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
unset DISPLAY
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 > "$out/gpu-series.csv" 2>&1 &
gpu_monitor=$!
vmstat -t 5 > "$out/cpu-series.txt" 2>&1 &
cpu_monitor=$!
cleanup() {
  kill "$gpu_monitor" "$cpu_monitor" 2>/dev/null || true
  wait "$gpu_monitor" "$cpu_monitor" 2>/dev/null || true
}
trap cleanup EXIT
set +e
timeout --signal=TERM --kill-after=30 2400 "$root/.envs/embclip-robothor/bin/python" -u \
  vendor/embclip-allenact/allenact/main.py scripts/embclip/sjc_profile_smoke_config.py -b . \
  --seed 0 --config_kwargs '{"headless":true,"train_gpu_ids":[0,1],"num_train_processes":60}' \
  -c "$checkpoint" -o "$out/allenact" \
  > "$out/train.log" 2>&1
rc=$?
set -e
printf '{"return_code": %s}\n' "$rc" > "$out/exit.json"
exit "$rc"
