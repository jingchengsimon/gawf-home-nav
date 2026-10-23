"""Export a read-only SJC snapshot of the current light navigation curves.

Run on SJC from the repository root with the PointNav environment. Writes a tar
stream to stdout containing two plots and the exact scalar values used in them.
"""

from datetime import datetime
from io import BytesIO
import argparse
import json
import math
from pathlib import Path
import re
import sys
import tarfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path.cwd()
GROUPS = {
    "pointnav": {
        "title": "PointNav light: training episode metrics",
        "window": "Rolling 50 PPO updates; completed training episodes",
        "target": 1_000_000,
        "seeds": "seed 0",
        "runs": [
            ("LSTM2", ["nav-pointnav2020-gibsonv2-light-lstm2-formal1m-sjc-v1-seed0-20261020"]),
            ("GaWF2", ["nav-pointnav2020-gibsonv2-light-gawf2-formal1m-sjc-v1-seed0-20261020"]),
        ],
    },
    "embclip": {
        "title": "EmbCLIP light: training episode metrics",
        "window": "Completed training tasks aggregated every ~10k global frames",
        "target": 5_000_000,
        "seeds": "seeds 0 and 1",
        "runs": [
            ("GRU1 s0", ["nav-robothor-embclip-light-gru1-formal1m-sjc-v2-seed0-20261020",
                          "nav-robothor-embclip-light-gru1-formal5m-sjc-v1-seed0-20261021"]),
            ("GRU1 s1", ["nav-robothor-embclip-light-gru1-formal5m-sjc-v1-seed1-20261021"]),
            ("GaWF1 s0", ["nav-robothor-embclip-light-gawf1-formal1m-sjc-v1-seed0-20261020",
                           "nav-robothor-embclip-light-gawf1-formal5m-sjc-v1-seed0-20261021"]),
            ("GaWF1 s1", ["nav-robothor-embclip-light-gawf1-formal5m-sjc-v1-seed1-20261021"]),
            ("GRU3 s0", ["nav-robothor-embclip-light-gru3-formal1m-sjc-v2-seed0-20261020",
                          "nav-robothor-embclip-light-gru3-formal5m-sjc-v1-seed0-20261021"]),
            ("GRU3 s1", ["nav-robothor-embclip-light-gru3-formal5m-sjc-v1-seed1-20261021"]),
            ("GaWF3 s0", ["nav-robothor-embclip-light-gawf3-formal1m-sjc-v1-seed0-20261020",
                           "nav-robothor-embclip-light-gawf3-formal5m-sjc-v1-seed0-20261021"]),
            ("GaWF3 s1", ["nav-robothor-embclip-light-gawf3-formal5m-sjc-v1-seed1-20261021"]),
        ],
    },
}


def scalar(path, tag):
    if not path.is_file():
        raise FileNotFoundError(path)
    event = EventAccumulator(str(path), size_guidance={"scalars": 0})
    event.Reload()
    if tag not in event.Tags().get("scalars", []):
        raise ValueError(f"Missing scalar {tag}: {path}")
    rows = [{"step": int(row.step), "value": float(row.value)} for row in event.Scalars(tag)]
    if not rows or any(not math.isfinite(row["value"]) or not 0 <= row["value"] <= 1
                       for row in rows):
        raise ValueError(f"Empty or invalid probability series: {path} {tag}")
    if any(a["step"] >= b["step"] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"Non-increasing steps: {path} {tag}")
    return {"source": str(path.relative_to(ROOT)), "points": rows}


def embclip_log_series(path):
    """AllenAct's TRAIN line has the same means before TensorBoard flushes."""
    rows = {"success": [], "spl": []}
    for line in path.read_text().splitlines():
        step = re.search(r"\bTRAIN:\s+(\d+) rollout steps", line)
        if step is None:
            continue
        for metric in rows:
            value = re.search(r"\b" + metric + r"\s+(\S+)", line)
            if value is None:
                raise ValueError(f"Missing {metric} in TRAIN line: {path}")
            rows[metric].append({"step": int(step.group(1)), "value": float(value.group(1))})
    if not rows["success"]:
        raise ValueError(f"No complete TRAIN metric lines: {path}")
    for points in rows.values():
        if any(not math.isfinite(p["value"]) or not 0 <= p["value"] <= 1 for p in points):
            raise ValueError(f"Invalid TRAIN probability: {path}")
        if any(a["step"] >= b["step"] for a, b in zip(points, points[1:])):
            raise ValueError(f"Non-increasing TRAIN steps: {path}")
    return {metric: {"source": str(path.relative_to(ROOT)), "points": points}
            for metric, points in rows.items()}


def one_part(group, experiment_id):
    if group == "pointnav":
        base = ROOT / "logs" / experiment_id / "tensorboard"
        series = {}
        for metric in ("success", "spl"):
            files = list((base / f"metrics_{metric}").glob("events.out.tfevents.*"))
            if len(files) != 1:
                raise ValueError(f"Expected one {metric} event file for {experiment_id}: {files}")
            series[metric] = scalar(files[0], "metrics")
    else:
        base = ROOT / "results" / experiment_id / "allenact" / "tb"
        files = list(base.rglob("events.out.tfevents.*"))
        if len(files) != 1:
            raise ValueError(f"Expected one EmbCLIP event file for {experiment_id}: {files}")
        try:
            series = {metric: scalar(files[0], f"train-metrics/{metric}")
                      for metric in ("success", "spl")}
        except ValueError as exc:
            if "Missing scalar" not in str(exc):
                raise
            series = embclip_log_series(ROOT / "results" / experiment_id / "train.log")
    if [p["step"] for p in series["success"]["points"]] != [p["step"] for p in series["spl"]["points"]]:
        raise ValueError(f"SR/SPL step mismatch: {experiment_id}")
    return series


def one_run(group, label, experiment_ids):
    parts = [one_part(group, experiment_id) for experiment_id in experiment_ids]
    series = {}
    for metric in ("success", "spl"):
        sources = [part[metric]["source"] for part in parts]
        points = [point for part in parts for point in part[metric]["points"]]
        if any(a["step"] >= b["step"] for a, b in zip(points, points[1:])):
            raise ValueError(f"Overlapping or non-increasing continuation: {label} {metric}")
        series[metric] = {"sources": sources, "points": points}
    return {"label": label, "experiment_ids": experiment_ids, "series": series}


def embclip_summary(runs, grid_steps=50_000):
    """Compare seeds at the same frames, only where both have observations."""
    models = list(dict.fromkeys(run["label"].rsplit(" ", 1)[0] for run in runs))
    summaries = []
    for model in models:
        seeds = {run["label"].rsplit(" ", 1)[1]: run for run in runs
                 if run["label"].rsplit(" ", 1)[0] == model}
        if set(seeds) != {"s0", "s1"}:
            raise ValueError(f"Need both seeds for {model}: {sorted(seeds)}")
        metrics = {}
        for metric in ("success", "spl"):
            pair = [seeds[seed]["series"][metric]["points"] for seed in ("s0", "s1")]
            start = max(points[0]["step"] for points in pair)
            end = min(points[-1]["step"] for points in pair)
            first = math.ceil(start / grid_steps) * grid_steps
            last = math.floor(end / grid_steps) * grid_steps
            if first > last:
                raise ValueError(f"No shared {grid_steps}-frame grid for {model} {metric}")
            steps = np.arange(first, last + 1, grid_steps, dtype=np.int64)
            aligned = np.array([
                np.interp(steps, [p["step"] for p in points],
                          [p["value"] for p in points]) for points in pair
            ])
            metrics[metric] = {
                "steps": steps.tolist(),
                "mean": aligned.mean(axis=0).tolist(),
                "std": aligned.std(axis=0, ddof=0).tolist(),
                "n": 2,
            }
        summaries.append({"label": model, "seeds": ["s0", "s1"], "series": metrics})
    return summaries


def figure(group, spec, runs, summaries=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    families = list(dict.fromkeys(run["label"].split()[0] for run in runs))
    for axis, metric in zip(axes, ("success", "spl")):
        if group == "embclip":
            for summary in summaries:
                data = summary["series"][metric]
                x = np.asarray(data["steps"]) / 1000
                mean = np.asarray(data["mean"])
                std = np.asarray(data["std"])
                color = colors[families.index(summary["label"])]
                axis.plot(x, mean, linewidth=1.7, color=color, label=summary["label"])
                axis.fill_between(x, np.maximum(0, mean - std), np.minimum(1, mean + std),
                                  color=color, alpha=0.16, linewidth=0)
                axis.plot(x[-1], mean[-1], marker="o", markersize=4, color=color)
        else:
            for run in runs:
                points = run["series"][metric]["points"]
                x = [p["step"] / 1000 for p in points]
                y = [p["value"] for p in points]
                color = colors[families.index(run["label"].split()[0])]
                axis.plot(x, y, linewidth=1.3, color=color, label=run["label"])
                axis.plot(x[-1], y[-1], marker="o", markersize=4, color=color)
        axis.set_ylabel("Training SR" if metric == "success" else "Training SPL")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left", fontsize=8, ncol=2 if group == "embclip" else 1)
    axes[-1].set_xlabel("Global training frames (thousands)")
    subtitle = ("mean ± population std across seeds 0/1; shared observed range only"
                if group == "embclip" else spec["seeds"])
    fig.suptitle(spec["title"] + f" · {subtitle} · {spec['target'] // 1_000_000}M target\n" +
                 spec["window"] + "; no validation episodes", fontsize=10)
    output = {}
    for ext in ("png", "pdf"):
        handle = BytesIO()
        fig.savefig(handle, format=ext, dpi=180, bbox_inches="tight")
        output[f"{group}_training_sr_spl.{ext}"] = handle.getvalue()
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    snapshot = {"host": "sjc-remote", "collected_at": datetime.now().astimezone().isoformat(),
                "metric_scope": "train episodes, not validation",
                "groups": {}}
    files = {}
    for group, spec in GROUPS.items():
        runs = [one_run(group, label, experiment_ids) for label, experiment_ids in spec["runs"]]
        summaries = embclip_summary(runs) if group == "embclip" else None
        snapshot["groups"][group] = {"metric_window": spec["window"],
                                      "target_global_frames": spec["target"], "runs": runs}
        if summaries is not None:
            snapshot["groups"][group]["summary_method"] = {
                "seed_count": 2, "grid_global_frames": 50_000,
                "alignment": "linear interpolation within each seed's observed range; no extrapolation",
                "std_ddof": 0, "plotted_range": "intersection of both seeds",
            }
            snapshot["groups"][group]["summaries"] = summaries
        files.update(figure(group, spec, runs, summaries))
    files["metrics_snapshot.json"] = (json.dumps(snapshot, indent=2) + "\n").encode()
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = datetime.now().timestamp()
            archive.addfile(info, BytesIO(content))


if __name__ == "__main__":
    main()
