"""Run one registered single-GPU PointNav throughput case and summarize steady FPS."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) update: (\d+)\tenv-time: ([\d.]+)s\tpth-time: ([\d.]+)s\tframes: ([\d.]+)")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def ports():
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    args = parser.parse_args()
    manifest_path = ROOT / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    run = json.loads(manifest_path.read_text())
    budget = run["budget"]
    environments = budget["environments_per_gpu"]
    rollout = budget["rollout_steps_per_environment"]
    updates = budget["updates"]
    if (run["experiment_id"] != args.experiment_id or run["host"] != "sjc-remote"
            or run["mode"] != "train_smoke" or run["model"] != "lstm2"
            or run["seed"] != 0 or run["execution"]["physical_gpu_index"] != 0
            or (environments, rollout, updates) not in {(2, 64, 16), (4, 64, 8), (2, 128, 8)}
            or budget["target_global_frames"] != environments * rollout * updates):
        raise RuntimeError("Unexpected benchmark identity or budget")
    for relative, expected in run["source_files"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError("Source hash mismatch: " + relative)
    result = ROOT / "results" / args.experiment_id
    logs = ROOT / "logs" / args.experiment_id
    if result.exists() or logs.exists():
        raise RuntimeError("Benchmark leaf already exists")
    result.mkdir()
    logs.mkdir()
    snapshot = result / "config_snapshot"
    for relative in run["source_files"]:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    write_json(result / "manifest_snapshot.json", run)
    launch_port, trainer_port = ports()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONPATH=str(ROOT / "vendor/habitat-lab-2020")
               + os.pathsep + str(ROOT / "scripts"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo", MAGNUM_LOG="quiet",
               HABITAT_SIM_LOG="quiet", GLOG_minloglevel="2", MASTER_ADDR="127.0.0.1",
               MASTER_PORT=str(launch_port), POINTNAV_DDPPO_MASTER_PORT=str(trainer_port))
    command = [str(ROOT / ".envs/pointnav2020/bin/python"), "-u", "-m", "torch.distributed.launch",
               "--use_env", "--nproc_per_node", "1", "--master_addr", "127.0.0.1",
               "--master_port", str(launch_port), str(ROOT / "scripts/train_pointnav2020_light_bench.py"),
               args.experiment_id, "--updates", str(updates), "--model", "lstm2", "--seed", "0"]
    write_json(result / "command.json", command)
    started = time.monotonic()
    samples = []
    with (logs / "worker.log").open("w") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            output.write(line)
            output.flush()
            match = LINE.search(line)
            if match:
                stamp, update, env_time, pth_time, frames = match.groups()
                samples.append(dict(timestamp=stamp, update=int(update), env_time_s=float(env_time),
                                    pth_time_s=float(pth_time), global_frames=int(float(frames))))
                write_json(result / "progress.json", dict(status="running", global_frames=int(float(frames)),
                           target_global_frames=budget["target_global_frames"], updates_completed=int(update) + 1))
        return_code = process.wait()
    complete_path = result / "rank_0.complete.json"
    complete = json.loads(complete_path.read_text()) if complete_path.is_file() else None
    summary = dict(experiment_id=args.experiment_id, variant=f"{environments}x{rollout}",
                   target_global_frames=budget["target_global_frames"], return_code=return_code,
                   elapsed_s=time.monotonic() - started, samples=samples,
                   complete_receipt=complete)
    steady = [sample for sample in samples if sample["update"] >= 1]
    if len(steady) >= 2:
        first, last = steady[0], steady[-1]
        seconds = (datetime.strptime(last["timestamp"], "%Y-%m-%d %H:%M:%S,%f")
                   - datetime.strptime(first["timestamp"], "%Y-%m-%d %H:%M:%S,%f")).total_seconds()
        frames = last["global_frames"] - first["global_frames"]
        summary["steady"] = dict(frames=frames, seconds=seconds, global_frames_per_second=frames / seconds,
                                 environment_time_s=last["env_time_s"] - first["env_time_s"],
                                 pytorch_time_s=last["pth_time_s"] - first["pth_time_s"])
    valid = (return_code == 0 and complete is not None and complete["status"] == "complete"
             and len(samples) >= 3 and samples[-1]["global_frames"] == budget["target_global_frames"]
             and "steady" in summary)
    summary["status"] = "complete" if valid else "failed"
    write_json(result / "summary.json", summary)
    run["status"] = summary["status"]
    run["units"][0]["status"] = summary["status"]
    run["completion_evidence"] = dict(summary_path=str(result / "summary.json"),
                                      global_frames=samples[-1]["global_frames"] if samples else 0,
                                      return_code=return_code,
                                      steady_global_frames_per_second=summary.get("steady", {}).get("global_frames_per_second"))
    write_json(manifest_path, run)
    subprocess.run([str(ROOT / ".envs/pointnav2020/bin/python"), "-B",
                    str(ROOT / "scripts/rebuild_index.py")], cwd=ROOT, check=True)
    if not valid:
        raise RuntimeError("Benchmark did not produce complete timed updates; inspect result and log")
    print(json.dumps(dict(variant=summary["variant"], steady=summary["steady"])), flush=True)


if __name__ == "__main__":
    main()
