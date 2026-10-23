"""Render one frozen snapshot of finished EmbCLIP RoboTHOR val evaluations."""

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experiments/monitoring/jobs"
ORDER = {"gru1": 0, "gawf1": 1, "gru3": 2, "gawf3": 3}
COLORS = {"gru1": "#1f77b4", "gawf1": "#ff7f0e", "gru3": "#2ca02c", "gawf3": "#d62728"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot():
    rows = []
    statuses = []
    for manifest in sorted(JOBS.glob("nav-robothor-embclip-light-*-eval-val1800-sjc-v1-seed[01]-20261022.json")):
        run = json.loads(manifest.read_text())
        model = run["model"]["family"]
        seed = run["seed"]
        if model not in ORDER or seed not in (0, 1):
            raise ValueError(f"Unexpected evaluation identity: {manifest}")
        statuses.append(dict(model=model, seed=seed, status=run["status"],
                             experiment_id=run["experiment_id"]))
        if run["status"] not in ("complete", "partial"):
            continue
        summary_path = ROOT / "results" / run["experiment_id"] / "summary.json"
        summary = json.loads(summary_path.read_text())
        metrics_path = ROOT / summary["metrics_path"]
        count = summary["episode_count"]
        missing = summary["missing_episode_ids"]
        sr = summary["success_rate"]
        spl = summary["spl"]
        if (summary["experiment_id"] != run["experiment_id"]
                or summary["status"] != run["status"]
                or summary["expected_episode_count"] != 1800
                or count + len(missing) != 1800
                or count != summary["unique_episode_count"]
                or not math.isclose(sr, summary["success_count"] / count, abs_tol=1e-12)
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in (sr, spl))
                or sha256(metrics_path) != summary["metrics_sha256"]):
            raise ValueError(f"Invalid evaluation evidence: {run['experiment_id']}")
        rows.append(dict(experiment_id=run["experiment_id"], model=model, seed=seed,
                         status=run["status"], episode_count=count,
                         missing_episode_ids=missing, success_count=summary["success_count"],
                         success_rate=sr, spl=spl, checkpoint_sha256=summary["checkpoint_sha256"],
                         metrics_path=summary["metrics_path"],
                         metrics_sha256=summary["metrics_sha256"]))
    if len(statuses) != 8 or not rows:
        raise ValueError("No finished validation evaluations found")
    rows.sort(key=lambda row: (ORDER[row["model"]], row["seed"]))
    statuses.sort(key=lambda row: (ORDER[row["model"]], row["seed"]))
    if any(row["missing_episode_ids"] != rows[0]["missing_episode_ids"] for row in rows):
        raise ValueError("Finished runs do not cover the same episode IDs")
    return rows, statuses


def render(rows, output):
    count = rows[0]["episode_count"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor("white")
    models = list(ORDER)
    grouped = {model: [row for row in rows if row["model"] == model] for model in models}
    for ax, key, heading in zip(axes, ("success_rate", "spl"), ("Validation SR", "Validation SPL")):
        all_values = [row[key] for row in rows]
        upper = max(0.21 if key == "success_rate" else 0.125, max(all_values) + 0.04)
        for x, model in enumerate(models):
            seed_rows = grouped[model]
            if not seed_rows:
                ax.text(x, 0.012, "pending", ha="center", va="bottom", fontsize=9, color="#333333")
                continue
            value = mean(row[key] for row in seed_rows)
            ax.bar(x, value, width=0.65, color=COLORS[model], alpha=0.82, zorder=2)
            for row in seed_rows:
                ax.scatter(x + (-0.11 if row["seed"] == 0 else 0.11), row[key],
                           marker="o" if row["seed"] == 0 else "D", s=56,
                           facecolors="white", edgecolors="#142638", linewidths=1.25, zorder=4)
            ax.text(x, max(row[key] for row in seed_rows) + 0.007,
                    f"{100 * value:.2f}%", ha="center", va="bottom", fontsize=9.5, color="#142638")
        ax.set_xlim(-0.55, len(models) - 0.45)
        ax.set_ylim(0, upper)
        ax.set_xticks(range(len(models)),
                      labels=[f"{model.upper()}\n(n={len(grouped[model])})" for model in models])
        ax.set_title(heading, fontsize=13, pad=12)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(axis="both", labelsize=9.5)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"EmbCLIP light · RoboTHOR validation · {count:,}/1,800 episodes",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.93, "Bars: mean of finished seeds  |  ○ seed 0  ◆ seed 1  |  5.001M training frames",
             ha="center", fontsize=9.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.81, bottom=0.18, wspace=0.18)
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/embclip-val1800-seed01-20261022")
    args = parser.parse_args()
    rows, statuses = snapshot()
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    now = datetime.now().astimezone()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"mean_scatter_{now.strftime('%Y%m%d_%H%M%S')}"
    model_summaries = []
    for model in ORDER:
        seed_rows = [row for row in rows if row["model"] == model]
        model_summaries.append(dict(model=model, completed_seed_count=len(seed_rows),
                                    seeds=[row["seed"] for row in seed_rows],
                                    mean_success_rate=mean(row["success_rate"] for row in seed_rows) if seed_rows else None,
                                    mean_spl=mean(row["spl"] for row in seed_rows) if seed_rows else None))
    payload = dict(observed_at=now.isoformat(), split="RoboTHOR ObjectNav val",
                   expected_episode_count=1800, common_episode_count=rows[0]["episode_count"],
                   finished_run_count=len(rows), plot_script_sha256=sha256(Path(__file__)),
                   evaluation_statuses=statuses, model_summaries=model_summaries, runs=rows)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    render(rows, output)
    print(output)


if __name__ == "__main__":
    main()
