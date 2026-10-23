#!/bin/bash
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the gawf-home-nav project root}"
: "${SLURM_JOB_GPUS:?GPU allocation is required}"
[[ "${1:-}" =~ ^nav-[a-z0-9]+(-[a-z0-9]+)+$ ]]
cd "$SLURM_SUBMIT_DIR"
mkdir -p "logs/$1"
sampler_pid=''
cleanup() {
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nvidia-smi --id="$SLURM_JOB_GPUS" \
  --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv --loop=1 >"logs/$1/gpu_memory.csv" 2>"logs/$1/gpu_memory.stderr" &
sampler_pid=$!
bash scripts/pointnav2020_job.sh "$1"
