#!/usr/bin/env bash
# Run two disjoint evaluation shards; preserve every failed attempt.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ID="${1:?Complete experiment ID required}"
ATTEMPT="${2:?Attempt leaf required}"
[[ "$EXPERIMENT_ID" =~ ^nav-pointnav-[a-z0-9-]+$ ]] || exit 2
[[ "$ATTEMPT" =~ ^attempt[0-9]+$ ]] || exit 2
RESULT="$ROOT/results/$EXPERIMENT_ID/$ATTEMPT"
mkdir -p "$(dirname "$RESULT")"
mkdir "$RESULT" || exit 2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
PYTHON="$ROOT/.envs/habitat-pointnav/bin/python"
CHECKPOINT="$ROOT/assets/downloads/gibson-2plus-resnet50.pth"
cd "$ROOT/vendor/habitat-lab" || exit 2
run_shard() {
    local rank="$1"
    "$PYTHON" -B "$ROOT/scripts/eval_pointnav.py" \
        --checkpoint "$CHECKPOINT" --output "$RESULT/gpu$rank" \
        --rank "$rank" --world-size 2 --episodes 5 --seed 0 --videos 2 \
        > "$RESULT/gpu$rank.log" 2>&1
    local code="$?"
    printf '{"exit_code":%d}\n' "$code" > "$RESULT/gpu$rank.exit.json"
    return "$code"
}
run_shard 0 & pid0=$!
run_shard 1 & pid1=$!
printf '{"gpu0_pid":%d,"gpu1_pid":%d}\n' "$pid0" "$pid1" > "$RESULT/processes.json"
wait "$pid0"; code0=$?
wait "$pid1"; code1=$?
printf '{"gpu0_exit":%d,"gpu1_exit":%d}\n' "$code0" "$code1" > "$RESULT/exit.json"
[[ "$code0" -eq 0 && "$code1" -eq 0 ]]
