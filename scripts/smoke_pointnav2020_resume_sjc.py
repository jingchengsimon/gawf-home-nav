"""On SJC, interrupt one GaWF formal-path update and resume the same run."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from run_pointnav2020_formal import child_trainers
from pointnav2020_runtime import ROOT

EXPERIMENT_ID = "nav-pointnav2020-gibsonv2-ddppo-gawf2-resume-smoke-sjc-v1-seed0-20261020"
TARGET = 768


def ports():
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


def launch(label, stop_after_first=False):
    launch_port, trainer_port = ports()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "vendor/habitat-lab-2020") + os.pathsep + str(ROOT / "scripts"),
               GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="1", SLURM_JOB_ID="910104", MASTER_ADDR="127.0.0.1",
               MASTER_PORT=str(launch_port), POINTNAV_DDPPO_MASTER_PORT=str(trainer_port))
    command = [str(ROOT / ".envs/pointnav2020/bin/python"), "-u", "-B", "-m", "torch.distributed.launch",
               "--use_env", "--nproc_per_node", "1", "--master_addr", "127.0.0.1",
               "--master_port", str(launch_port), str(ROOT / "scripts/train_pointnav2020.py"),
               EXPERIMENT_ID, "--formal", "--model", "gawf2", "--seed", "0",
               "--target-steps", str(TARGET), "--checkpoint-steps", "256"]
    logs = ROOT / "logs" / EXPERIMENT_ID
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / (label + ".log")).open("w") as output:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        signaled = False
        deadline = time.monotonic() + 1200
        while child.poll() is None and time.monotonic() < deadline:
            path = ROOT / "results" / EXPERIMENT_ID / "progress.json"
            if stop_after_first and not signaled and path.exists():
                progress = json.loads(path.read_text())
                if progress.get("saved_global_frames", 0) >= 256:
                    ranks = child_trainers(child.pid)
                    if ranks:
                        for pid in ranks:
                            os.kill(pid, signal.SIGTERM)
                        signaled = True
            time.sleep(1)
        if child.poll() is None:
            child.terminate()
            raise TimeoutError(label)
        return child.returncode, signaled


def main():
    results = ROOT / "results" / EXPERIMENT_ID
    if results.exists():
        raise FileExistsError(results)
    first_rc, signaled = launch("first", stop_after_first=True)
    assert signaled and first_rc != 0, (signaled, first_rc)
    checkpoint = results / "checkpoints/interrupted.pth"
    import torch
    state = torch.load(checkpoint, map_location="cpu")
    before = state["requeue_stats"]["count_steps"]
    assert before < TARGET and state["optim_state"]["state"]
    second_rc, _ = launch("resumed")
    progress = json.loads((results / "progress.json").read_text())
    assert second_rc == 0 and progress["global_frames"] >= TARGET
    assert progress["global_frames"] > before
    assert progress["updates_completed"] > state["requeue_stats"]["start_update"]
    (results / "resume_smoke.json").write_text(json.dumps(
        dict(status="pass", first_return_code=first_rc, saved_before_resume=before,
             global_frames_after_resume=progress["global_frames"],
             updates_after_resume=progress["updates_completed"], optimizer_state_restored=True), indent=2) + "\n")
    print((results / "resume_smoke.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
