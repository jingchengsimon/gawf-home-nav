"""Evaluate explicitly registered RoboTHOR 5M checkpoints for any seed 0–4."""

import argparse
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experiments/monitoring/jobs"
MODELS = {"gru1", "gawf1", "gru3", "gawf3"}
EVAL_COMPLETE = re.compile(r"\bnum_test_tasks_complete\s+(\d+)")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def completed_from_log_line(line):
    match = EVAL_COMPLETE.search(line)
    return int(match.group(1)) if match else None


def run_logged_evaluation(command, output, env, expected_episodes):
    progress_path = output / "progress.json"
    write_json(progress_path, dict(status="running", completed_episodes=0,
               expected_episodes=expected_episodes,
               updated_at=datetime.now().astimezone().isoformat()))
    completed = 0
    with (output / "eval.log").open("w") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace",
                              bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                observed = completed_from_log_line(line)
                if observed is not None and completed < observed <= expected_episodes:
                    completed = observed
                    write_json(progress_path, dict(status="running", completed_episodes=completed,
                               expected_episodes=expected_episodes,
                               updated_at=datetime.now().astimezone().isoformat()))
            return_code = process.wait()
    if return_code != 0:
        write_json(progress_path, dict(status="failed", completed_episodes=completed,
                   expected_episodes=expected_episodes,
                   updated_at=datetime.now().astimezone().isoformat()))
    return return_code


def validation_ids(files):
    ids, scenes = set(), set()
    count = 0
    for name, expected_hash in sorted(files.items()):
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"Validation file missing or changed: {name}")
        with gzip.open(path, "rt") as stream:
            episodes = json.load(stream)
        for episode in episodes:
            ids.add(episode["id"])
            scenes.add(episode["scene"])
            count += 1
    if len(files) != 15 or count != 1800 or len(ids) != 1800 or len(scenes) != 15:
        raise RuntimeError("Expected 1,800 unique episodes in 15 validation scenes")
    return ids, scenes


def finalize_metrics(path, run, output, expected_ids, expected_scenes):
    source = run["source_checkpoint"]
    exit_record = json.loads((output / "exit.json").read_text())
    if exit_record["return_code"] != 0:
        raise RuntimeError("AllenAct did not exit cleanly")
    metrics_files = list((output / "allenact/metrics").rglob("metrics__test_*.json"))
    if len(metrics_files) != 1:
        raise RuntimeError(f"Expected one AllenAct metrics file, found {len(metrics_files)}")
    metrics_path = metrics_files[0]
    metrics = json.loads(metrics_path.read_text())
    if len(metrics) != 1 or int(metrics[0]["training_steps"]) != 5_001_216:
        raise RuntimeError("Metrics checkpoint step mismatch")
    tasks = metrics[0]["tasks"]
    observed_ids = [item["task_info"]["id"] for item in tasks]
    observed_scenes = {item["task_info"]["scene"] for item in tasks}
    if (not tasks or len(observed_ids) != len(set(observed_ids))
            or not set(observed_ids) <= expected_ids
            or not observed_scenes <= expected_scenes):
        raise RuntimeError("Invalid validation episode identities")
    missing_ids = sorted(expected_ids - set(observed_ids))
    status = "partial" if missing_ids else "complete"
    success_count = sum(bool(item["success"]) for item in tasks)
    sr = success_count / len(tasks)
    spl = sum(float(item["spl"]) for item in tasks) / len(tasks)
    if abs(sr - metrics[0]["success"]) > 1e-6 or abs(spl - metrics[0]["spl"]) > 1e-6:
        raise RuntimeError("Aggregate SR/SPL does not match per-episode results")
    summary = dict(status=status, experiment_id=run["experiment_id"],
                   split="robothor_objectnav_val_full_attempt",
                   expected_episode_count=1800, episode_count=len(tasks),
                   unique_episode_count=len(tasks), scene_count=len(observed_scenes),
                   missing_episode_ids=missing_ids, missing_episode_count=len(missing_ids),
                   success_count=success_count, success_rate=sr, spl=spl,
                   mean_episode_length=metrics[0]["ep_length"],
                   checkpoint_steps=5_001_216, checkpoint_sha256=source["sha256"],
                   metrics_path=str(metrics_path.relative_to(ROOT)),
                   metrics_sha256=sha256(metrics_path),
                   ai2thor_version="5.0.0",
                   ai2thor_commit="f0825767cd50d69f666c7f282e54abfe58f1e917")
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", dict(status=status,
               completed_episodes=len(tasks), expected_episodes=run["expected_episodes"],
               updated_at=datetime.now().astimezone().isoformat()))
    run["status"] = run["units"][0]["status"] = status
    run.pop("failure", None)
    run["completion_evidence"] = dict(summary_path=str((output / "summary.json").relative_to(ROOT)),
                                       episode_count=len(tasks), missing_episode_count=len(missing_ids),
                                       success_count=success_count, success_rate=sr, spl=spl,
                                       metrics_sha256=summary["metrics_sha256"], return_code=0)
    write_json(path, run)
    print(f"{status.upper()} {run['experiment_id']} {len(tasks)}/1800 SR={sr:.6f} SPL={spl:.6f}", flush=True)


def validate_manifest(run, path, gpu):
    model, seed = run["model"]["family"], run["seed"]
    if (run["host"] != "sjc-remote" or run["mode"] != "checkpoint_eval"
            or run["experiment_id"] != path.stem or model not in MODELS
            or seed not in range(5) or (seed > 1 and model not in {"gru1", "gawf1"})
            or run["expected_episodes"] != 1800
            or gpu not in (0, 1) or run["execution"]["physical_gpu_index"] != gpu
            or run["execution"].get("evaluation_seed", 0) != 0
            or run["execution"].get("num_test_processes", 15) != 15
            or run["execution"].get("test_on_validation", True) is not True):
        raise RuntimeError(f"Unexpected evaluation manifest: {path}")
    if run["status"] not in {"prepared", "queued", "failed"}:
        raise RuntimeError("Evaluation must be explicitly prepared or queued")
    runner = str(Path(__file__).relative_to(ROOT))
    digest = sha256(Path(__file__))
    if run.get("runner_path") != runner or run.get("runner_sha256") != digest:
        raise RuntimeError("Manifest must pin the unified evaluation runner")
    if run["source"]["source_files"].get(runner) != digest:
        raise RuntimeError("Runner is missing from the pinned evaluation sources")


def run_one(path, gpu, validate_only=False):
    run = json.loads(path.read_text())
    validate_manifest(run, path, gpu)
    model, seed = run["model"]["family"], run["seed"]
    source = run["source_checkpoint"]
    train_path = JOBS / (source["training_experiment_id"] + ".json")
    train = json.loads(train_path.read_text())
    evidence = train.get("completion_evidence") or {}
    if (train["status"] != "complete" or train["model"]["family"] != model
            or train["execution"]["seed"] != seed
            or evidence.get("checkpoint_path") != source["path"]
            or evidence.get("checkpoint_sha256") != source["sha256"]
            or evidence.get("checkpoint_steps") != 5_001_216):
        raise RuntimeError(f"Training checkpoint identity mismatch: {train_path}")
    checkpoint = ROOT / source["path"]
    if not checkpoint.is_file() or sha256(checkpoint) != source["sha256"]:
        raise RuntimeError(f"Training checkpoint missing or changed: {checkpoint}")
    for name, digest in run["source"]["source_files"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Evaluation config changed: {name}")
    expected_ids, expected_scenes = validation_ids(run["validation"]["files"])
    output = ROOT / run["units"][0]["result_path"]
    if output.parent.resolve() != (ROOT / "results").resolve() or output.is_symlink():
        raise RuntimeError(f"Unsafe result path: {output}")
    if output.exists():
        raise RuntimeError(f"Result leaf already exists: {output}; use a fresh registered ID")
    python = ROOT / ".envs/embclip-robothor/bin/python"
    if not python.is_file():
        raise RuntimeError(f"Evaluation environment missing: {python}")
    import torch
    state = torch.load(checkpoint, map_location="cpu")
    required = {"model_state_dict", "optimizer_state_dict", "training_pipeline_state_dict",
                "trainer_seed", "scheduler_state"}
    if required - state.keys() or int(state["total_steps"]) != 5_001_216 or int(state["trainer_seed"]) != seed:
        raise RuntimeError("Checkpoint state/seed/step validation failed")
    del state
    train_result = ROOT / train["units"][0]["result_path"]
    if "process exiting with exitcode 0" not in (train_result / "train.log").read_text(errors="replace")[-16384:]:
        raise RuntimeError("No clean training exit in log")
    exit_path = train_result / "exit.json"
    if exit_path.exists() and json.loads(exit_path.read_text())["return_code"] != 0:
        raise RuntimeError("Training exit receipt is nonzero")
    import shutil
    if shutil.disk_usage(ROOT / "results").free < 5 * 1024**3:
        raise RuntimeError("Less than 5 GiB available")
    uuid, free = subprocess.check_output(["nvidia-smi", "--id="+str(gpu),
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    if int(free) < 11000 or any(line.split(",")[0].strip() == uuid.strip() for line in apps.splitlines()):
        raise RuntimeError("Target GPU is occupied or has less than 11000 MiB free")
    if validate_only:
        print(json.dumps(dict(status="validated", experiment_id=run["experiment_id"], seed=seed,
              checkpoint_sha256=source["sha256"], actual_frames=5001216, gpu=gpu)), flush=True)
        return
    output.mkdir()
    run["status"] = run["units"][0]["status"] = "running"
    write_json(path, run)
    command = [
        "timeout", "--signal=TERM", "--kill-after=30", "14400",
        str(python), "-u", "vendor/embclip-allenact/allenact/main.py",
        f"scripts/embclip/light5m_{model}_config.py", "-b", ".",
        "--eval", "--seed", "0", "--config_kwargs",
        json.dumps(dict(headless=True, test_on_validation=True,
                        test_gpu_ids=[0], num_test_processes=15)),
        "-c", str(checkpoint), "-o", str(output / "allenact"),
    ]
    env = os.environ.copy()
    env.update(HOME=str(ROOT / "assets/embclip-robothor/home"),
               PATH=str(ROOT / "assets/embclip-robothor/vulkan-tools/package/usr/bin") + ":" + env["PATH"],
               PYTHONPATH=str(ROOT / "vendor/embclip-allenact") + ":" + str(ROOT),
               CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("DISPLAY", None)
    write_json(output / "launch.json", dict(command=command, physical_gpu=gpu,
               checkpoint_sha256=source["sha256"],
               started_at=datetime.now().astimezone().isoformat()))
    try:
        return_code = run_logged_evaluation(command, output, env, run["expected_episodes"])
        write_json(output / "exit.json", dict(return_code=return_code,
                   finished_at=datetime.now().astimezone().isoformat()))
        if return_code != 0:
            raise RuntimeError(f"AllenAct evaluation exited {return_code}: {output / 'eval.log'}")
        finalize_metrics(path, run, output, expected_ids, expected_scenes)
    except Exception as error:
        run["status"] = run["units"][0]["status"] = "failed"
        run["failure"] = str(error)
        write_json(path, run)
        progress_path = output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(
            completed_episodes=0, expected_episodes=1800)
        progress.update(status="failed", updated_at=datetime.now().astimezone().isoformat())
        write_json(progress_path, progress)
        raise
    finally:
        subprocess.run([sys.executable, "-B", "scripts/rebuild_index.py"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_ids", nargs="+", help="Explicit registered evaluation IDs, in queue order")
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    paths = []
    for experiment_id in args.experiment_ids:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
            raise RuntimeError("Expected an experiment ID, not a path")
        path = JOBS / (experiment_id + ".json")
        run = json.loads(path.read_text())
        validate_manifest(run, path, args.gpu)
        if path in paths:
            raise RuntimeError("Duplicate evaluation ID")
        paths.append(path)
    for path in paths:
        run_one(path, args.gpu, args.validate_only)


if __name__ == "__main__":
    main()
