"""Run one registered SJC-only light PointNav training smoke on one GPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def available_ports():
    with socket.socket() as launcher, socket.socket() as trainer:
        launcher.bind(("127.0.0.1", 0))
        trainer.bind(("127.0.0.1", 0))
        return launcher.getsockname()[1], trainer.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    args = parser.parse_args()
    manifest_path = ROOT / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["experiment_id"] == args.experiment_id
    assert manifest["host"] == "sjc-remote" and manifest["mode"] == "train-smoke"
    assert manifest["resources"]["gpus"] == 1
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "1"}
    for relative, digest in manifest["source_files"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == digest, "Source mismatch: " + relative
    results = ROOT / "results" / args.experiment_id
    logs = ROOT / "logs" / args.experiment_id
    assert not results.exists() and not logs.exists(), "Preserve prior attempt; use a new ID"
    results.mkdir(parents=True)
    logs.mkdir(parents=True)
    (results / "checkpoints").mkdir()
    launch_port, trainer_port = available_ports()
    env = dict(os.environ,
               PYTHONPATH=str(ROOT / "vendor/habitat-lab-2020") + os.pathsep + str(ROOT / "scripts"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo",
               MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", GLOG_minloglevel="2",
               MASTER_ADDR="127.0.0.1", MASTER_PORT=str(launch_port),
               POINTNAV_DDPPO_MASTER_PORT=str(trainer_port))
    budget = manifest["budget"]
    command = [str(ROOT / ".envs/pointnav2020/bin/python"), "-u", "-m", "torch.distributed.launch",
               "--use_env", "--nproc_per_node", "1", "--master_addr", "127.0.0.1",
               "--master_port", str(launch_port), str(ROOT / "scripts/train_pointnav2020_light.py"),
               args.experiment_id, "--formal", "--model", manifest["model"], "--seed", str(manifest["seed"]),
               "--target-steps", str(budget["target_global_frames"]),
               "--checkpoint-steps", str(budget["checkpoint_global_frames"])]
    (results / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with (logs / "worker.log").open("w") as output:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, timeout=budget["runtime_seconds"])
    progress_path = results / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    success = (completed.returncode == 0 and
               progress.get("global_frames", 0) >= budget["target_global_frames"] and
               (results / "rank_0.complete.json").exists() and
               Path(progress.get("checkpoint_path", "missing")).exists())
    receipt = dict(experiment_id=args.experiment_id, return_code=completed.returncode,
                   success=success, elapsed_s=time.monotonic() - started,
                   progress=progress, gpu=os.environ["CUDA_VISIBLE_DEVICES"])
    (results / "smoke_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not success:
        raise RuntimeError("Light PointNav smoke failed; see logs/worker.log and receipt")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
