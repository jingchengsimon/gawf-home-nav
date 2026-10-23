"""Run one registered SJC EmbCLIP job, resuming its latest full checkpoint."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
STEP_RE = re.compile(r"__steps_(\d+)\.pt$")
MODELS = {"gru1", "gru3", "lstm3"}
CHILD = None


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_step(path):
    match = STEP_RE.search(path.name)
    return int(match.group(1)) if match else -1


def verify_checkpoint(path, target):
    import torch

    state = torch.load(path, map_location="cpu")
    required = {"model_state_dict", "optimizer_state_dict",
                "training_pipeline_state_dict", "trainer_seed", "scheduler_state"}
    missing = required - state.keys()
    if missing:
        raise RuntimeError(f"Incomplete checkpoint {path}: {sorted(missing)}")
    steps = int(state["total_steps"])
    if steps != checkpoint_step(path) or steps > target:
        raise RuntimeError(f"Checkpoint step mismatch or beyond target: {path}, {steps}, {target}")
    del state
    return steps


def on_signal(signum, _frame):
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signum)


def main():
    global CHILD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    args = parser.parse_args()
    manifest_path = ROOT / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    run = json.loads(manifest_path.read_text())
    execution = run["execution"]
    model = run["model"]["family"]
    target = int(execution["steps_budget"])
    gpu = int(execution["physical_gpu_index"])
    interval = int(execution["checkpoint_interval_steps"])
    if (run["experiment_id"] != args.experiment_id or run["host"] != "sjc-remote"
            or model not in MODELS or gpu not in (0, 1) or target < 2048
            or interval != 100_000 or execution["global_train_envs"] != 16):
        raise RuntimeError("Unexpected job identity or training protocol")
    for relative, expected in run["source"]["source_files"].items():
        source = ROOT / relative
        if not source.is_file() or sha256(source) != expected:
            raise RuntimeError(f"Source hash mismatch: {relative}")
    output = ROOT / run["units"][0]["result_path"]
    if not output.is_dir() or not (output / "config_snapshot").is_dir():
        raise RuntimeError(f"Missing registered result and config snapshot: {output}")
    for relative, expected in run["source"]["source_files"].items():
        snapshot = output / "config_snapshot" / relative
        if not snapshot.is_file() or sha256(snapshot) != expected:
            raise RuntimeError(f"Config snapshot mismatch: {snapshot}")
    if not (ROOT / "data/embclip-robothor/robothor-objectnav/train/episodes/FloorPlan_Train1_1.json.gz").is_file():
        raise RuntimeError("Training episodes unavailable")
    if not (ROOT / "assets/embclip-robothor/home/.cache/clip/RN50.pt").is_file():
        raise RuntimeError("CLIP RN50 unavailable")
    python = ROOT / ".envs/embclip-robothor/bin/python"
    if not python.is_file():
        raise RuntimeError("EmbCLIP environment unavailable")

    local_checkpoints = [p for p in (output / "allenact/checkpoints").rglob("*.pt")
                         if checkpoint_step(p) >= 0]
    checkpoint = max(local_checkpoints, key=checkpoint_step) if local_checkpoints else None
    if checkpoint is None and run.get("resume_from"):
        prior = run["resume_from"]
        checkpoint = ROOT / prior["checkpoint_path"]
        if not checkpoint.is_file() or sha256(checkpoint) != prior["sha256"]:
            raise RuntimeError("Pinned continuation checkpoint missing or changed")
    starting_steps = verify_checkpoint(checkpoint, target) if checkpoint else 0
    if starting_steps >= target:
        write_json(output / "summary.json", dict(status="complete", global_frames=starting_steps,
                    checkpoint_path=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha256(checkpoint)))
        return

    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    number = 1
    while (attempts / f"{number:03d}").exists():
        number += 1
    attempt = attempts / f"{number:03d}"
    attempt.mkdir()
    config_kwargs = dict(headless=True, train_gpu_ids=[0], num_train_processes=16,
                         target_steps=target, checkpoint_interval=interval)
    command = [str(python), "-u", "vendor/embclip-allenact/allenact/main.py",
               f"scripts/embclip/light_{model}_config.py", "-b", ".",
               "--seed", "0", "--config_kwargs", json.dumps(config_kwargs),
               "-o", str(output / "allenact")]
    if checkpoint:
        command += ["--checkpoint", str(checkpoint)]
    env = os.environ.copy()
    env.update(HOME=str(ROOT / "assets/embclip-robothor/home"),
               PATH=str(ROOT / "assets/embclip-robothor/vulkan-tools/package/usr/bin") + ":" + env["PATH"],
               PYTHONPATH=str(ROOT / "vendor/embclip-allenact") + ":" + str(ROOT),
               CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("DISPLAY", None)
    write_json(attempt / "launch.json", dict(started_at=datetime.now().astimezone().isoformat(),
               starting_steps=starting_steps,
               checkpoint_path=str(checkpoint.relative_to(ROOT)) if checkpoint else None,
               checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
               target_steps=target, checkpoint_interval=interval, physical_gpu=gpu,
               command=command))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)
    with (attempt / "train.log").open("wb") as attempt_log, (output / "train.log").open("ab") as combined:
        CHILD = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
        for line in CHILD.stdout:
            attempt_log.write(line)
            combined.write(line)
            combined.flush()
        rc = CHILD.wait()
    checkpoints = [p for p in (output / "allenact/checkpoints").rglob("*.pt")
                   if checkpoint_step(p) >= 0]
    latest = max(checkpoints, key=checkpoint_step) if checkpoints else None
    latest_steps = verify_checkpoint(latest, target) if latest else starting_steps
    write_json(attempt / "exit.json", dict(return_code=rc, latest_steps=latest_steps,
               finished_at=datetime.now().astimezone().isoformat()))
    write_json(output / "exit.json", dict(return_code=rc, latest_steps=latest_steps,
               attempt=number, finished_at=datetime.now().astimezone().isoformat()))
    if rc == 0 and latest_steps >= target:
        write_json(output / "summary.json", dict(status="complete", global_frames=latest_steps,
                   checkpoint_path=str(latest.relative_to(ROOT)), checkpoint_sha256=sha256(latest)))
    if rc != 0 or latest_steps < target:
        raise RuntimeError(f"Training stopped at {latest_steps}/{target} steps; rerun this exact ID to resume")


if __name__ == "__main__":
    main()
