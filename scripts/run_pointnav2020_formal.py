"""Launch one registered four-GPU PointNav formal run or requeued continuation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def child_trainers(launcher_pid):
    rows = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    children = {}
    commands = {}
    for row in rows.splitlines():
        parts = row.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            pid, parent = int(parts[0]), int(parts[1])
            children.setdefault(parent, []).append(pid)
            commands[pid] = parts[2]
    pending, descendants = [launcher_pid], []
    while pending:
        pid = pending.pop()
        pending.extend(children.get(pid, []))
        if pid != launcher_pid:
            descendants.append(pid)
    return [pid for pid in descendants if "train_pointnav2020.py" in commands.get(pid, "")]


def available_ports():
    with socket.socket() as launcher, socket.socket() as trainer:
        launcher.bind(("127.0.0.1", 0))
        trainer.bind(("127.0.0.1", 0))
        return launcher.getsockname()[1], trainer.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    args = parser.parse_args()
    path = ROOT / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    manifest = json.loads(path.read_text())
    assert manifest["experiment_id"] == args.experiment_id
    assert manifest["host"] == "amarel" and manifest["mode"] == "formal_train"
    assert manifest["resources"]["gpus"] == 4 and os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("SLURM_JOB_GPUS"), "A four-GPU Slurm allocation is required"
    for relative, digest in manifest["source_files"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == digest, "Formal source mismatch: " + relative
    results = ROOT / "results" / args.experiment_id
    logs = ROOT / "logs" / args.experiment_id
    results.mkdir(parents=True, exist_ok=True)
    (results / "checkpoints").mkdir(exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    assert not (results / "summary.json").exists(), "Completed run must not restart"
    attempt = "%s-restart%s" % (os.environ["SLURM_JOB_ID"], os.environ.get("SLURM_RESTART_COUNT", "0"))
    attempt_log = logs / ("worker.%s.log" % attempt)
    assert not attempt_log.exists(), "Existing attempt log must be preserved"
    launch_port, trainer_port = available_ports()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "vendor/habitat-lab-2020") + os.pathsep + str(ROOT / "scripts"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo",
               MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", GLOG_minloglevel="2",
               MASTER_ADDR="127.0.0.1", MASTER_PORT=str(launch_port),
               POINTNAV_DDPPO_MASTER_PORT=str(trainer_port))
    python = str(ROOT / ".envs/pointnav2020/bin/python")
    budget = manifest["budget"]
    command = [python, "-u", "-m", "torch.distributed.launch", "--use_env", "--nproc_per_node", "4",
               "--master_addr", "127.0.0.1", "--master_port", str(launch_port),
               str(ROOT / "scripts/train_pointnav2020.py"), args.experiment_id,
               "--formal", "--model", manifest["model"], "--seed", str(manifest["seed"]),
               "--target-steps", str(budget["target_global_frames"]),
               "--checkpoint-steps", str(budget["checkpoint_global_frames"])]
    (results / ("command.%s.json" % attempt)).write_text(json.dumps(command, indent=2) + "\n")
    pending_signal = False

    def request_requeue(signum, frame):
        nonlocal pending_signal
        pending_signal = True

    signal.signal(signal.SIGUSR1, request_requeue)
    started = time.monotonic()
    with attempt_log.open("w") as output:
        worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        signaled = set()
        while worker.poll() is None:
            if pending_signal:
                for pid in child_trainers(worker.pid):
                    if pid not in signaled:
                        try:
                            os.kill(pid, signal.SIGUSR1)
                        except ProcessLookupError:
                            continue
                        signaled.add(pid)
            time.sleep(1)
        return_code = worker.returncode
    progress_path = results / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    complete = (return_code == 0 and progress.get("global_frames", 0) >= budget["target_global_frames"]
                and all((results / ("rank_%d.complete.json" % rank)).exists() for rank in range(4)))
    if complete:
        summary = dict(status="complete", experiment_id=args.experiment_id,
                       global_frames=progress["global_frames"],
                       updates_completed=progress["updates_completed"],
                       checkpoint_path=progress["checkpoint_path"],
                       slurm_job_id=os.environ["SLURM_JOB_ID"],
                       elapsed_s=time.monotonic() - started)
        (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (results / ("exit.%s.json" % attempt)).write_text(json.dumps(
        dict(return_code=return_code, complete=complete, signal_forwarded=bool(signaled),
             saved_global_frames=progress.get("saved_global_frames"),
             elapsed_s=time.monotonic() - started), indent=2) + "\n")
    if not complete:
        raise RuntimeError("Formal segment ended before target; inspect log and resume checkpoint")


if __name__ == "__main__":
    main()
