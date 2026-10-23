"""Plot the completed RGB-only PointNav seed0 validation comparison."""

from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pointnav-rgb-seed0-validation-20261023"
MODELS = (
    ("lstm2", "LSTM2", "#1f77b4"),
    ("gawf2", "GaWF2", "#ff7f0e"),
)
SCENES = ["Cantwell", "Eastville", "Edgemere", "Scioto"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = {}
    for model, _, _ in MODELS:
        eval_id = f"nav-pointnav2020-gibsonv2-rgb-light-{model}-evalval284-sjc-v1-seed0-20261023"
        summary_path = OUT / "sources" / f"{model}.summary.json"
        exit_path = OUT / "sources" / f"{model}.exit.json"
        summary = json.loads(summary_path.read_text())
        receipt = json.loads(exit_path.read_text())
        manifest = json.loads((ROOT / "experiments/monitoring/jobs" / f"{eval_id}.json").read_text())
        train = json.loads((ROOT / "experiments/monitoring/jobs" /
                            f"{summary['source_experiment_id']}.json").read_text())
        assert receipt["return_code"] == 0
        assert manifest["status"] == summary["status"] == train["status"] == "complete"
        assert summary["experiment_id"] == manifest["experiment_id"] == eval_id
        assert manifest["policy_observations"] == ["rgb", "pointgoal"]
        assert manifest["seed"] == 0 and manifest["expected_episodes"] == 284
        assert summary["completed_episodes"] == 284 and summary["validation_scenes"] == SCENES
        assert summary["checkpoint_global_frames"] == 1_000_064
        assert summary["checkpoint_sha256"] == manifest["checkpoint_sha256"]
        assert summary["checkpoint_sha256"] == train["completion_evidence"]["checkpoint_sha256"]
        assert all(0 <= summary["metrics"][key] <= 1 for key in ("success", "spl"))
        data[model] = dict(summary=summary, summary_sha256=sha256(summary_path),
                           exit_sha256=sha256(exit_path))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    for ax, metric, title, ymax, digits in (
        (axes[0], "success", "Validation SR", 0.45, 3),
        (axes[1], "spl", "Validation SPL", 0.035, 4),
    ):
        for x, (model, label, color) in enumerate(MODELS):
            value = 100 * data[model]["summary"]["metrics"][metric]
            ax.bar(x, value, width=0.56, color=color, alpha=0.88, zorder=2)
            ax.scatter(x, value, s=52, facecolor="white", edgecolor="black",
                       linewidth=1.15, zorder=4, clip_on=False)
            annotation = f"{value:.{digits}f}%"
            if metric == "success":
                count = round(data[model]["summary"]["metrics"]["success"] * 284)
                annotation += f"  ({count}/284)"
            ax.annotate(annotation, (x, value), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xticks((0, 1), [label for _, label, _ in MODELS])
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Score (%)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(axis="y", color="#d9dee5", linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
    fig.suptitle("RGB-only PointNav · Gibson v2 validation · seed 0", fontsize=15)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"pointnav_rgb_seed0_validation_sr_spl.{extension}",
                    dpi=180, facecolor="white")
    plt.close(fig)
    snapshot = dict(created_at=datetime.now().astimezone().isoformat(),
                    protocol="pointnav2020-gibsonv2-light-rgb-fixedgoal-resnet18-v1",
                    dataset="Gibson PointNav v2 val", scenes=SCENES,
                    episodes_per_model=284, plot="mean bar plus seed scatter",
                    models=data)
    (OUT / "validation_scores.json").write_text(json.dumps(snapshot, indent=2) + "\n")


if __name__ == "__main__":
    main()
