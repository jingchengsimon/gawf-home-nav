#!/bin/bash
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the gawf-home-nav project root}"
cd "$SLURM_SUBMIT_DIR"
exec .envs/pointnav2020/bin/python -u -B scripts/run_pointnav2020_smoke.py "$1"
