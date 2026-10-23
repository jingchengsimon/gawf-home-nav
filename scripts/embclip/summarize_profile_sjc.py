"""Verify and summarize the bounded SJC EmbCLIP throughput profile."""
import csv
import glob
import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/nav-robothor-embclip-profile-smoke-sjc-v1-seed0-20261020"
PRIOR = ROOT / "results/nav-robothor-embclip-resume-smoke-sjc-v1-seed0-20261019"
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def only_checkpoint(directory):
    paths = list(directory.glob("allenact/checkpoints/**/*.pt"))
    assert len(paths) == 1, paths
    return paths[0]


def main():
    assert json.loads((RUN / "exit.json").read_text())["return_code"] == 0
    log = re.sub(r"\x1b\[[0-9;]*m", "", (RUN / "train.log").read_text())
    assert log.count("Training finished successfully.") == 2
    assert log.count("Loading checkpoint from ") >= 2
    points = []
    for line in log.splitlines():
        if "TRAIN: " not in line:
            continue
        stamp = re.search(r"\[(\d\d/\d\d \d\d:\d\d:\d\d)", line)
        step = re.search(r"TRAIN: (\d+) rollout steps", line)
        assert stamp and step, line
        point = {"time": "2026/" + stamp.group(1), "steps": int(step.group(1))}
        for key in ("reward", "success", "spl", "total_loss", "elapsed_time"):
            match = re.search(r"\b" + key + r" (\S+)", line)
            assert match, (key, line)
            point[key] = float(match.group(1).rstrip("s"))
            assert math.isfinite(point[key])
        points.append(point)
    assert [p["steps"] for p in points] == [30720, 46080, 61440, 76800]
    log_times = re.findall(r"\[(\d\d/\d\d \d\d:\d\d:\d\d)", log)
    assert log_times
    profile_wall_seconds = (
        datetime.strptime("2026/" + log_times[-1], TIME_FORMAT)
        - datetime.strptime("2026/" + log_times[0], TIME_FORMAT)
    ).total_seconds()
    start = datetime.strptime(points[0]["time"], TIME_FORMAT)
    end = datetime.strptime(points[-1]["time"], TIME_FORMAT)
    intervals = []
    for before, after in zip(points, points[1:]):
        seconds = (datetime.strptime(after["time"], TIME_FORMAT)
                   - datetime.strptime(before["time"], TIME_FORMAT)).total_seconds()
        intervals.append({
            "from_steps": before["steps"], "to_steps": after["steps"],
            "seconds": seconds,
            "global_steps_per_second": (after["steps"] - before["steps"]) / seconds,
        })
    gpu = {0: {"util": [], "mem": []}, 1: {"util": [], "mem": []}}
    with (RUN / "gpu-series.csv").open() as handle:
        for row in csv.DictReader(handle):
            time = datetime.strptime(row["timestamp"].strip().split(".")[0], TIME_FORMAT)
            if start <= time <= end:
                index = int(row[" index"].strip())
                gpu[index]["util"].append(int(row[" utilization.gpu [%]"].strip().split()[0]))
                gpu[index]["mem"].append(int(row[" memory.used [MiB]"].strip().split()[0]))
    gpu_stats = {
        str(index): {
            "samples": len(values["util"]),
            "mean_utilization_percent": statistics.mean(values["util"]),
            "max_utilization_percent": max(values["util"]),
            "mean_memory_mib": statistics.mean(values["mem"]),
            "max_memory_mib": max(values["mem"]),
        }
        for index, values in gpu.items()
    }
    cpu = []
    for line in (RUN / "cpu-series.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 20 or not parts[0].isdigit():
            continue
        time = datetime.strptime(parts[-2] + " " + parts[-1], "%Y-%m-%d %H:%M:%S")
        if start <= time <= end:
            cpu.append({"user": int(parts[12]), "system": int(parts[13]),
                        "idle": int(parts[14])})
    assert cpu and all(row["samples"] for row in gpu_stats.values())
    old_path, new_path = only_checkpoint(PRIOR), only_checkpoint(RUN)
    old = torch.load(old_path, map_location="cpu")
    new = torch.load(new_path, map_location="cpu")
    assert old["total_steps"] == 15360 and new["total_steps"] == 76800
    assert len(new["optimizer_state_dict"]["state"]) == 17
    assert all(torch.isfinite(value).all().item()
               for value in new["model_state_dict"].values())
    changed = sum(
        not torch.equal(old["model_state_dict"][key], new["model_state_dict"][key])
        for key in old["model_state_dict"]
    )
    assert changed == 17
    seconds = (end - start).total_seconds()
    steady = (points[-1]["steps"] - points[0]["steps"]) / seconds
    summary = {
        "status": "complete", "experiment_id": RUN.name,
        "execution_host": "sjc-remote", "exit_code": 0, "completed_stages": 1,
        "world_size": 2, "num_train_processes_global": 60,
        "global_frames": 76800, "resumed_from_global_frames": 15360,
        "new_global_frames": 61440,
        "profile_wall_seconds_from_log": profile_wall_seconds,
        "last_log_interval_seconds": points[-1]["elapsed_time"],
        "steady_window": {
            "from_steps": points[0]["steps"], "to_steps": points[-1]["steps"],
            "seconds": seconds, "global_steps_per_second": steady,
            "synchronized_effective_steps_per_gpu_second": steady / 2,
            "intervals": intervals,
        },
        "train_metric_points": points,
        "gpu_steady_window": gpu_stats,
        "cpu_steady_window": {
            "samples": len(cpu),
            "mean_user_percent": statistics.mean(row["user"] for row in cpu),
            "mean_system_percent": statistics.mean(row["system"] for row in cpu),
            "mean_idle_percent": statistics.mean(row["idle"] for row in cpu),
        },
        "input_checkpoint_sha256": sha256(old_path),
        "output_checkpoint_path": str(new_path.relative_to(ROOT)),
        "output_checkpoint_sha256": sha256(new_path),
        "output_checkpoint_bytes": new_path.stat().st_size,
        "optimizer_state_entries": 17, "changed_model_tensors": changed,
        "model_tensors_all_finite": True, "ai2thor_version": "5.0.0",
        "ai2thor_commit": "f0825767cd50d69f666c7f282e54abfe58f1e917",
    }
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
