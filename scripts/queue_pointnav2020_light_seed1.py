"""Launch two registered SJC PointNav seed1 runs after both seed0 runs complete."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor/habitat-lab-2020"), str(ROOT / "scripts")]
DATE = "20261021"
MODELS = {"lstm2": 0, "gawf2": 1}
SOURCE = {model: f"nav-pointnav2020-gibsonv2-light-{model}-formal1m-sjc-v1-seed0-20261020"
          for model in MODELS}
TARGET = {model: f"nav-pointnav2020-gibsonv2-light-{model}-formal1m-sjc-v1-seed1-{DATE}"
          for model in MODELS}
RECEIPT = ROOT / "experiments/deployment" / f"sjc-pointnav-light-seed1-gate-{DATE}.json"


def read_manifest(experiment_id):
    path = ROOT / "experiments/monitoring/jobs" / (experiment_id + ".json")
    value = json.loads(path.read_text())
    if value["experiment_id"] != experiment_id:
        raise RuntimeError(f"Manifest identity mismatch: {path}")
    return path, value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def session_alive(session):
    panes = subprocess.run(["tmux", "list-panes", "-t", session, "-F", "#{pane_dead}"],
                           capture_output=True, text=True)
    return panes.returncode == 0 and any(pane == "0" for pane in panes.stdout.splitlines())


def source_ready(model):
    experiment_id = SOURCE[model]
    _, run = read_manifest(experiment_id)
    if run["status"] != "complete":
        return None
    session = run["execution"]["session"]
    if session_alive(session):
        return None
    processes = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                               text=True, check=True).stdout
    if any(experiment_id in line for line in processes.splitlines()):
        return None
    result = ROOT / "results" / experiment_id
    summary_path = result / "summary.json"
    progress_path = result / "progress.json"
    exit_receipts = sorted(result.glob("exit.attempt*.json"))
    checkpoint = result / "checkpoints/interrupted.pth"
    rank_complete = result / "rank_0.complete.json"
    if not all(p.is_file() for p in (summary_path, progress_path, checkpoint, rank_complete)) or not exit_receipts:
        raise RuntimeError(f"Completed manifest lacks full result evidence: {experiment_id}")
    summary = json.loads(summary_path.read_text())
    progress = json.loads(progress_path.read_text())
    receipt = json.loads(exit_receipts[-1].read_text())
    target = run["budget"]["target_global_frames"]
    digest = sha256(checkpoint)
    if not (summary["status"] == "complete" and summary["experiment_id"] == experiment_id
            and summary["global_frames"] >= target
            and progress["global_frames"] >= target and progress["saved_global_frames"] >= target
            and receipt["return_code"] == 0 and receipt["complete"]
            and digest == summary["checkpoint_sha256"]
            == run["completion_evidence"]["checkpoint_sha256"]):
        raise RuntimeError(f"Seed0 completion evidence disagrees: {experiment_id}")
    import torch
    state = torch.load(checkpoint, map_location="cpu")
    required = {"state_dict", "optim_state", "lr_sched_state", "requeue_stats", "formal_meta"}
    if required - set(state) or state["formal_meta"] != dict(
            model=model, seed=0, target_steps=target,
            checkpoint_steps=run["budget"]["checkpoint_global_frames"]) or int(
                state["requeue_stats"]["count_steps"]) != summary["global_frames"]:
        raise RuntimeError(f"Seed0 checkpoint is incomplete or mismatched: {experiment_id}")
    return dict(experiment_id=experiment_id, global_frames=summary["global_frames"],
                checkpoint_sha256=digest, summary_path=str(summary_path.relative_to(ROOT)))


def validate_target(model):
    path, run = read_manifest(TARGET[model])
    source_runs = [read_manifest(SOURCE[m])[1] for m in MODELS]
    reference = next(item for item in source_runs if item["model"] == model)
    if not (run["host"] == "sjc-remote" and run["mode"] == "formal_train"
            and run["model"] == model and run["seed"] == 1
            and run["budget"] == reference["budget"]
            and run["source_files"] == reference["source_files"]
            and run["resume_from"] is None
            and run["execution"]["physical_gpu_index"] == MODELS[model]
            and run["execution"]["launch_after"] == [SOURCE[m] for m in MODELS]):
        raise RuntimeError(f"Seed1 protocol mismatch: {TARGET[model]}")
    for relative, digest in run["source_files"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Training source changed: {relative}")
    if run["status"] not in ("queued", "running"):
        raise RuntimeError(f"Unexpected seed1 status: {run['status']}")
    return path, run


def start_target(model):
    path, run = validate_target(model)
    experiment_id = TARGET[model]
    session = f"pointnav-light-{model}-1m-seed1-{DATE}"
    if run["status"] == "running":
        if not session_alive(session):
            raise RuntimeError(f"Seed1 manifest says running without tmux: {experiment_id}")
        return session
    if not session_alive(session):
        if (ROOT / "results" / experiment_id).exists() or (ROOT / "logs" / experiment_id).exists():
            raise RuntimeError(f"Seed1 result/log leaf already exists: {experiment_id}")
        launcher_log = ROOT / "experiments/deployment" / (experiment_id + ".launcher.log")
        if launcher_log.exists():
            raise RuntimeError(f"Seed1 launcher log already exists: {launcher_log}")
        command = [str(ROOT / ".envs/pointnav2020/bin/python"), "-B",
                   str(ROOT / "scripts/run_pointnav2020_light_train.py"), experiment_id]
        shell = ("CUDA_VISIBLE_DEVICES=" + str(MODELS[model]) + " "
                 + " ".join(shlex.quote(part) for part in command)
                 + " > " + shlex.quote(str(launcher_log)) + " 2>&1")
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT), shell],
                       check=True)
    if not session_alive(session):
        raise RuntimeError(f"Seed1 tmux did not remain active: {experiment_id}")
    run["status"] = "running"
    run["units"][0]["status"] = "running"
    run["execution"]["session"] = session
    run["execution"]["activated_at"] = datetime.now().astimezone().isoformat()
    write_json(path, run)
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate setup and report the current gate without waiting")
    args = parser.parse_args()
    for model in MODELS:
        validate_target(model)
    if args.check:
        print(json.dumps({"gate": "both_seed0_complete", "source_status": {
            model: read_manifest(SOURCE[model])[1]["status"] for model in MODELS},
            "target_status": {model: read_manifest(TARGET[model])[1]["status"] for model in MODELS}}))
        return
    while True:
        if all(read_manifest(SOURCE[model])[1]["status"] == "complete" for model in MODELS):
            break
        time.sleep(30)
    evidence = {model: source_ready(model) for model in MODELS}
    while not all(evidence.values()):
        time.sleep(30)
        evidence = {model: source_ready(model) for model in MODELS}
    activated = {}
    for model in MODELS:
        activated[model] = dict(experiment_id=TARGET[model], session=start_target(model))
    subprocess.run([str(ROOT / ".envs/pointnav2020/bin/python"), "-B",
                    str(ROOT / "scripts/rebuild_index.py")], cwd=ROOT, check=True)
    write_json(RECEIPT, dict(gate="both_seed0_complete", source_evidence=evidence,
                             activated=activated, activated_at=datetime.now().astimezone().isoformat()))
    print(json.dumps(dict(activated=activated, source_evidence=evidence)), flush=True)


if __name__ == "__main__":
    main()
