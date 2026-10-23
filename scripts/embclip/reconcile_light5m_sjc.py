"""Finalize completed SJC EmbCLIP 5M runs whose legacy wrapper rejected rollout overshoot."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experiments/monitoring/jobs"
CHECKPOINT_STEP = re.compile(r"__steps_(\d+)\.pt$")
MODELS = {"gru1", "gawf1", "gru3", "gawf3"}
SEEDS = {0, 1, 2, 3, 4}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidates():
    for path in sorted(JOBS.glob("nav-robothor-embclip-light-*-formal5m-sjc-v1-seed*.json")):
        run = json.loads(path.read_text())
        execution = run.get("execution") or {}
        model = run.get("model") or {}
        if (run.get("experiment_id") != path.stem or run.get("host") != "sjc-remote"
                or run.get("mode") != "formal_train" or model.get("family") not in MODELS
                or execution.get("seed") not in SEEDS or execution.get("steps_budget") != 5_000_000
                or execution.get("global_train_envs") != 16 or execution.get("rollout") != 128):
            continue
        if run.get("status") == "running":
            yield path, run


def active_sessions():
    output = subprocess.run(["tmux", "list-panes", "-a", "-F",
                             "#{session_name}|#{pane_dead}"], capture_output=True,
                            text=True, check=False).stdout
    return {parts[0] for line in output.splitlines()
            if len(parts := line.split("|", 1)) == 2 and parts[1] == "0"}


def active_processes():
    return subprocess.run(["ps", "-eo", "args"], capture_output=True,
                          text=True, check=True).stdout.splitlines()


def reconcile(path, run, sessions, processes):
    experiment_id = run["experiment_id"]
    execution = run["execution"]
    session = execution.get("session")
    if session in sessions or execution.get("queue_session") in sessions:
        return False
    if any(experiment_id in command for command in processes):
        return False
    result = ROOT / run["units"][0]["result_path"]
    if result.parent.resolve() != (ROOT / "results").resolve() or result.is_symlink():
        raise RuntimeError(f"Unexpected result path: {result}")
    checkpoint_root = result / "allenact/checkpoints"
    checkpoints = [(int(match.group(1)), item) for item in checkpoint_root.rglob("*.pt")
                   if (match := CHECKPOINT_STEP.search(item.name))]
    if not checkpoints:
        return False
    steps, checkpoint = max(checkpoints)
    target = int(execution["steps_budget"])
    rollout_size = int(execution["global_train_envs"]) * int(execution["rollout"])
    if not target <= steps <= target + rollout_size:
        return False
    log = result / "train.log"
    if not log.is_file() or b"process exiting with exitcode 0" not in log.read_bytes()[-16384:]:
        return False
    exit_path = result / "exit.json"
    if exit_path.is_file() and json.loads(exit_path.read_text()).get("return_code") != 0:
        return False

    import torch
    state = torch.load(checkpoint, map_location="cpu")
    required = {"model_state_dict", "optimizer_state_dict",
                "training_pipeline_state_dict", "trainer_seed", "scheduler_state"}
    if required - state.keys() or int(state["total_steps"]) != steps:
        raise RuntimeError(f"Incomplete or mismatched checkpoint: {checkpoint}")
    if int(state["trainer_seed"]) != int(execution["seed"]):
        raise RuntimeError(f"Checkpoint seed mismatch: {checkpoint}")
    del state
    digest = sha256(checkpoint)
    relative_checkpoint = str(checkpoint.relative_to(ROOT))
    evidence = dict(global_frames=steps, checkpoint_steps=steps,
                    checkpoint_path=relative_checkpoint, checkpoint_sha256=digest,
                    clean_training_log_exit_observed=True,
                    wrapper_finalization="legacy rollout overshoot rejected after training",
                    observed_at=datetime.now().astimezone().isoformat())
    write_json(result / "summary.json", dict(status="complete", global_frames=steps,
                                             checkpoint_path=relative_checkpoint,
                                             checkpoint_sha256=digest,
                                             reconciliation=evidence))
    run["status"] = "complete"
    run["units"][0]["status"] = "complete"
    run["completion_evidence"] = evidence
    write_json(path, run)
    print(f"completed {experiment_id} at {steps:,} frames", flush=True)
    return True


def pass_once():
    sessions = active_sessions()
    processes = active_processes()
    changed = False
    for path, run in candidates():
        changed = reconcile(path, run, sessions, processes) or changed
    if changed:
        subprocess.run([sys.executable, "scripts/rebuild_index.py"], cwd=ROOT, check=True)
    return sum(1 for _ in candidates())


if __name__ == "__main__":
    watch = len(sys.argv) == 2 and sys.argv[1] == "--watch"
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and not watch):
        raise SystemExit("usage: reconcile_light5m_sjc.py [--watch]")
    while True:
        remaining = pass_once()
        if not watch or remaining == 0:
            break
        time.sleep(60)
