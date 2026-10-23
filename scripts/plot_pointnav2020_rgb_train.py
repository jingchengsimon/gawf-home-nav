"""Plot the RGB-only PointNav seed0 training scalar snapshot."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pointnav-rgb-seed0-validation-20261023"
COLORS = {"LSTM2": "#1f77b4", "GaWF2": "#ff7f0e"}


def main():
    data = json.loads((OUT / "sources/training_scalars.json").read_text())
    assert data["protocol"] == "pointnav2020-gibsonv2-light-rgb-fixedgoal-resnet18-v1"
    assert [run["label"] for run in data["runs"]] == ["LSTM2", "GaWF2"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                             constrained_layout=True)
    for ax, metric, label in zip(axes, ("success", "spl"), ("Training SR", "Training SPL")):
        for run in data["runs"]:
            model = run["label"]
            expected_id = ("nav-pointnav2020-gibsonv2-rgb-light-"
                           f"{'lstm2' if model == 'LSTM2' else 'gawf2'}-formal1m-sjc-v1-seed0-20261022")
            assert run["experiment_ids"] == [expected_id]
            series = run["series"][metric]
            points = series["points"]
            assert points and all(0 <= p["value"] <= 1 for p in points)
            assert all(a["step"] < b["step"] for a, b in zip(points, points[1:]))
            assert points[-1]["step"] <= 1_000_064
            x = [p["step"] / 1000 for p in points]
            y = [p["value"] for p in points]
            ax.plot(x, y, color=COLORS[model], linewidth=1.35, label=model)
            ax.scatter(x[-1], y[-1], s=25, color=COLORS[model], zorder=4)
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.24)
        ax.legend(frameon=False, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Global training frames (thousands)")
    fig.suptitle("RGB-only PointNav · Gibson v2 training · seed 0\n"
                 "Completed training episodes · rolling 50 PPO updates", fontsize=13)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"pointnav_rgb_seed0_training_sr_spl.{extension}",
                    dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
