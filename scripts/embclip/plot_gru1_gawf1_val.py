"""Collect verified SJC validation scores and plot selected GRU1/GaWF1 seeds.

Run ``collect`` on SJC with this file supplied on stdin; run ``render`` locally.
The collector reads experiment outputs only and writes its JSON snapshot to stdout.
"""

import gzip
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path


MODELS = ("gru1", "gawf1")
SEEDS_BY_MODEL = {"gru1": (0, 1, 2, 3, 4), "gawf1": (0, 1, 3)}
COLORS = {"gru1": "#1f77b4", "gawf1": "#ff7f0e"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def average(values):
    return sum(values) / len(values)


def collect():
    root = Path.cwd()
    episode_files = sorted((root / "data/embclip-robothor/robothor-objectnav/val/episodes").glob("*.json.gz"))
    expected = {episode["id"] for path in episode_files for episode in json.load(gzip.open(path, "rt"))}
    if len(episode_files) != 15 or len(expected) != 1800:
        raise RuntimeError("Expected 15 validation scenes and 1,800 unique episodes")

    rows = []
    for model in MODELS:
        for seed in SEEDS_BY_MODEL[model]:
            date = "20261022" if seed < 2 else "20261023"
            eid = f"nav-robothor-embclip-light-{model}-eval-val1800-sjc-v1-seed{seed}-{date}"
            manifest_path = root / "experiments/monitoring/jobs" / f"{eid}.json"
            if not manifest_path.exists():
                raise RuntimeError(f"Requested evaluation is missing: {eid}")
            manifest = json.loads(manifest_path.read_text())
            if manifest["status"] not in ("complete", "partial"):
                raise RuntimeError(f"Requested evaluation is unfinished: {eid}")
            summary_path = root / "results" / eid / "summary.json"
            summary = json.loads(summary_path.read_text())
            evidence = manifest["completion_evidence"]
            if (manifest["experiment_id"] != eid or manifest["model"]["family"] != model
                    or manifest["seed"] != seed or evidence["return_code"] != 0
                    or summary["experiment_id"] != eid or summary["status"] != manifest["status"]
                    or summary["checkpoint_sha256"] != manifest["source_checkpoint"]["sha256"]
                    or summary["checkpoint_steps"] != 5001216
                    or summary["expected_episode_count"] != 1800
                    or summary["ai2thor_version"] != "5.0.0"):
                raise RuntimeError(f"Invalid experiment identity or exit: {eid}")
            metrics_path = root / summary["metrics_path"]
            if digest(metrics_path) != summary["metrics_sha256"]:
                raise RuntimeError(f"Metrics hash mismatch: {eid}")
            metrics = json.loads(metrics_path.read_text())
            if len(metrics) != 1:
                raise RuntimeError(f"Unexpected metrics format: {eid}")
            tasks = {}
            for task in metrics[0]["tasks"]:
                task_id = task["task_info"]["id"]
                if task_id in tasks:
                    raise RuntimeError(f"Duplicate episode: {eid} {task_id}")
                tasks[task_id] = (float(task["success"]), float(task["spl"]))
            if (not set(tasks) <= expected or len(tasks) != summary["episode_count"]
                    or len(tasks) != summary["unique_episode_count"]
                    or sorted(expected - set(tasks)) != summary["missing_episode_ids"]
                    or not math.isclose(average([v[0] for v in tasks.values()]), summary["success_rate"], abs_tol=1e-12)
                    or not math.isclose(average([v[1] for v in tasks.values()]), summary["spl"], abs_tol=1e-12)):
                raise RuntimeError(f"Episode coverage or score mismatch: {eid}")
            rows.append({"model": model, "seed": seed, "experiment_id": eid,
                         "status": summary["status"], "episode_count": len(tasks),
                         "summary_sha256": digest(summary_path),
                         "metrics_sha256": summary["metrics_sha256"],
                         "checkpoint_sha256": summary["checkpoint_sha256"],
                         "tasks": tasks})
    if len(rows) != sum(len(seeds) for seeds in SEEDS_BY_MODEL.values()):
        raise RuntimeError("Requested evaluation count mismatch")
    common = set.intersection(*(set(row["tasks"]) for row in rows))
    if not common:
        raise RuntimeError("No common validation episodes")
    for row in rows:
        tasks = row.pop("tasks")
        row["common_sr"] = average([tasks[eid][0] for eid in common])
        row["common_spl"] = average([tasks[eid][1] for eid in common])
    snapshot = {"observed_at": datetime.now().astimezone().isoformat(),
                "expected_episode_count": 1800, "common_episode_count": len(common),
                "common_episode_ids_sha256": hashlib.sha256("\n".join(sorted(common)).encode()).hexdigest(),
                "selected_seeds": {model: list(seeds) for model, seeds in SEEDS_BY_MODEL.items()},
                "runs": rows}
    print(json.dumps(snapshot, indent=2, sort_keys=True))


def render(snapshot_path, output_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    snapshot = json.loads(snapshot_path.read_text())
    rows = snapshot["runs"]
    expected = snapshot["selected_seeds"]
    for model in MODELS:
        if sorted(row["seed"] for row in rows if row["model"] == model) != expected[model]:
            raise RuntimeError(f"Snapshot does not contain all selected {model} seeds")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.25))
    for ax, metric, title, ylabel in zip(
        axes, ("common_sr", "common_spl"), ("Validation SR", "Validation SPL"),
        ("SR (%)", "SPL (%)")
    ):
        max_value = max(row[metric] for row in rows) * 100
        rng = np.random.default_rng(0)
        for x, model in enumerate(MODELS):
            selected = [row for row in rows if row["model"] == model]
            values = np.array([row[metric] * 100 for row in selected])
            sem = values.std(ddof=1) / np.sqrt(values.size)
            ax.bar(x, values.mean(), width=0.72, yerr=sem, color=COLORS[model],
                   edgecolor="none", capsize=2.5, zorder=2,
                   error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"})
            jitter = rng.uniform(-0.22 * 0.72, 0.22 * 0.72, size=values.size)
            ax.scatter(x + jitter, values, s=10, color="#333333", alpha=0.52,
                       linewidths=0, zorder=3)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, max_value + 5)
        ax.set_xticks((0, 1), ("GRU1", "GaWF1"), rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, pad=12)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("EmbCLIP RoboTHOR validation", fontsize=14, y=0.97)
    fig.text(0.5, 0.89,
             f"GRU1: {len(expected['gru1'])} seeds  ·  GaWF1: {len(expected['gawf1'])} seeds  ·  {snapshot['common_episode_count']:,}/1,800 shared episodes",
             ha="center", fontsize=9.5)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.79, wspace=0.27)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "collect":
        collect()
    elif len(sys.argv) == 2 and sys.argv[1] == "collect-all":
        SEEDS_BY_MODEL = {model: (0, 1, 2, 3, 4) for model in MODELS}
        collect()
    elif len(sys.argv) == 4 and sys.argv[1] == "render":
        render(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit("usage: plot_gru1_gawf1_val.py collect | collect-all | render SNAPSHOT OUTPUT_STEM")
