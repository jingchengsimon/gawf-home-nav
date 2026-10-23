"""Run or restart one registered SJC light PointNav seed on one GPU.

A later manifest may use resume_from to extend a completed run into a new leaf.
The source result, checkpoint, and optimizer state remain untouched.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ports():
    with socket.socket() as launcher, socket.socket() as trainer:
        launcher.bind(("127.0.0.1", 0))
        trainer.bind(("127.0.0.1", 0))
        return launcher.getsockname()[1], trainer.getsockname()[1]


def continuation(manifest, results):
    source = manifest.get("resume_from")
    if source is None:
        return
    prior_id = source["experiment_id"]
    assert prior_id != manifest["experiment_id"]
    prior_manifest = json.loads((ROOT / "experiments/monitoring/jobs" / (prior_id + ".json")).read_text())
    assert prior_manifest["status"] == "complete"
    assert (prior_manifest["host"], prior_manifest["model"], prior_manifest["seed"]) == (
        "sjc-remote", manifest["model"], manifest["seed"])
    for relative in ("configs/pointnav2020/light/task.yaml",
                     "configs/pointnav2020/light/ddppo.yaml",
                     "configs/pointnav2020/light/protocol.json",
                     "scripts/train_pointnav2020_light_longrun.py"):
        assert prior_manifest["source_files"][relative] == manifest["source_files"][relative]
    prior_results = ROOT / "results" / prior_id
    prior_summary = json.loads((prior_results / "summary.json").read_text())
    assert prior_summary["status"] == "complete"
    prior_checkpoint = prior_results / "checkpoints/interrupted.pth"
    digest = sha256(prior_checkpoint)
    assert digest == source["checkpoint_sha256"] == prior_summary["checkpoint_sha256"]
    import torch
    state = torch.load(prior_checkpoint, map_location="cpu")
    required = {"state_dict", "optim_state", "lr_sched_state", "requeue_stats", "formal_meta"}
    assert required <= set(state)
    old_meta = state["formal_meta"]
    assert old_meta == dict(model=manifest["model"], seed=manifest["seed"],
                            target_steps=prior_manifest["budget"]["target_global_frames"],
                            checkpoint_steps=prior_manifest["budget"]["checkpoint_global_frames"])
    old_frames = int(state["requeue_stats"]["count_steps"])
    assert old_frames >= old_meta["target_steps"]
    assert manifest["budget"]["target_global_frames"] > old_frames
    assert manifest["budget"]["checkpoint_global_frames"] == old_meta["checkpoint_steps"]
    state["formal_meta"] = dict(old_meta, target_steps=manifest["budget"]["target_global_frames"])
    state["continuation_from"] = dict(experiment_id=prior_id, checkpoint_sha256=digest,
                                       global_frames=old_frames)
    target = results / "checkpoints/interrupted.pth"
    assert not target.exists()
    temporary = target.with_suffix(".pth.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    (results / "continuation.json").write_text(json.dumps(dict(
        source_experiment_id=prior_id, source_checkpoint_sha256=digest,
        source_global_frames=old_frames, initial_checkpoint_sha256=sha256(target),
        target_global_frames=manifest["budget"]["target_global_frames"]), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    args = parser.parse_args()
    path = ROOT / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    manifest = json.loads(path.read_text())
    assert manifest["experiment_id"] == args.experiment_id
    assert manifest["host"] == "sjc-remote" and manifest["mode"] == "formal_train"
    assert manifest["resources"]["gpus"] == 1
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(manifest["execution"]["physical_gpu_index"])
    assert manifest["model"] in {"lstm2", "gawf2"} and manifest["seed"] >= 0
    assert manifest["budget"]["target_global_frames"] > 0
    assert manifest["budget"]["checkpoint_global_frames"] > 0
    for relative, digest in manifest["source_files"].items():
        assert sha256(ROOT / relative) == digest, "Source mismatch: " + relative
    results = ROOT / "results" / args.experiment_id
    logs = ROOT / "logs" / args.experiment_id
    spec = dict(experiment_id=args.experiment_id, model=manifest["model"], seed=manifest["seed"],
                budget=manifest["budget"], source_files=manifest["source_files"],
                resume_from=manifest.get("resume_from"))
    if results.exists():
        assert json.loads((results / "run_spec.json").read_text()) == spec
        assert not (results / "summary.json").exists(), "Completed run must not restart"
        assert (results / "checkpoints/interrupted.pth").exists(), "Restart requires full checkpoint"
    else:
        assert not logs.exists(), "Prior logs require a distinct experiment ID"
        (results / "checkpoints").mkdir(parents=True)
        logs.mkdir()
        (results / "run_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        continuation(manifest, results)
    attempt = len(list(logs.glob("worker.attempt*.log"))) + 1
    log_path = logs / ("worker.attempt%03d.log" % attempt)
    assert not log_path.exists()
    launch_port, trainer_port = ports()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "vendor/habitat-lab-2020") + os.pathsep + str(ROOT / "scripts"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo",
               MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", GLOG_minloglevel="2",
               MASTER_ADDR="127.0.0.1", MASTER_PORT=str(launch_port),
               POINTNAV_DDPPO_MASTER_PORT=str(trainer_port))
    budget = manifest["budget"]
    command = [str(ROOT / ".envs/pointnav2020/bin/python"), "-u", "-m", "torch.distributed.launch",
               "--use_env", "--nproc_per_node", "1", "--master_addr", "127.0.0.1",
               "--master_port", str(launch_port), str(ROOT / "scripts/train_pointnav2020_light_longrun.py"),
               args.experiment_id, "--formal", "--model", manifest["model"], "--seed", str(manifest["seed"]),
               "--target-steps", str(budget["target_global_frames"]),
               "--checkpoint-steps", str(budget["checkpoint_global_frames"])]
    (results / ("command.attempt%03d.json" % attempt)).write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with log_path.open("w") as output:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    progress_path = results / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    checkpoint = results / "checkpoints/interrupted.pth"
    success = (completed.returncode == 0 and
               progress.get("global_frames", 0) >= budget["target_global_frames"] and
               progress.get("saved_global_frames", 0) >= budget["target_global_frames"] and
               (results / "rank_0.complete.json").exists() and checkpoint.is_file())
    receipt = dict(return_code=completed.returncode, complete=success,
                   global_frames=progress.get("global_frames"),
                   saved_global_frames=progress.get("saved_global_frames"),
                   elapsed_s=time.monotonic() - started)
    (results / ("exit.attempt%03d.json" % attempt)).write_text(json.dumps(receipt, indent=2) + "\n")
    if not success:
        if checkpoint.is_file():
            manifest["status"] = "recovering"
            manifest["completion_evidence"] = dict(global_frames=progress.get("global_frames", 0),
                                                    saved_global_frames=progress.get("saved_global_frames", 0),
                                                    last_attempt=attempt)
            path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise RuntimeError("Run stopped before target; restart the same ID from its full checkpoint")
    summary = dict(status="complete", experiment_id=args.experiment_id,
                   global_frames=progress["global_frames"],
                   updates_completed=progress["updates_completed"],
                   checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                   elapsed_s=receipt["elapsed_s"], resume_from=manifest.get("resume_from"))
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["status"] = "complete"
    manifest["units"][0]["status"] = "complete"
    manifest["completion_evidence"] = dict(global_frames=summary["global_frames"],
                                            completed_updates=summary["updates_completed"],
                                            checkpoint_sha256=summary["checkpoint_sha256"],
                                            summary_path=str(results / "summary.json"))
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([str(ROOT / ".envs/pointnav2020/bin/python"), "-B",
                    str(ROOT / "scripts/rebuild_index.py")], cwd=ROOT, check=True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
