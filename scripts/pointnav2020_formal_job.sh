#!/bin/bash
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the project root}"
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${SLURM_JOB_GPUS:?Four allocated GPUs are required}"
[[ "${1:-}" =~ ^nav-[a-z0-9]+(-[a-z0-9]+)+$ ]]
cd "$SLURM_SUBMIT_DIR"
mkdir -p "logs/$1"
driver_pid=''
sampler_pid=''
attempt="${SLURM_JOB_ID}-restart${SLURM_RESTART_COUNT:-0}"
nvidia-smi --id="$SLURM_JOB_GPUS" \
  --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv --loop=10 >"logs/$1/gpu.$attempt.csv" 2>"logs/$1/gpu.$attempt.stderr" &
sampler_pid=$!
cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
}
forward_signal() {
  if [[ -n "$driver_pid" ]]; then
    kill -USR1 "$driver_pid" 2>/dev/null || true
  fi
}
trap forward_signal USR1
trap cleanup EXIT
.envs/pointnav2020/bin/python -u -B scripts/run_pointnav2020_formal.py "$1" &
driver_pid=$!
while true; do
  set +e
  wait "$driver_pid"
  result=$?
  set -e
  if kill -0 "$driver_pid" 2>/dev/null; then
    continue
  fi
  exit "$result"
done
