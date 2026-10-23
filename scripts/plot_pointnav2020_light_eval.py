"""Export the paired PointNav seed0 Gibson-v2 validation scores as a tar stream."""

from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
MODELS = ("lstm2", "gawf2")
IDS = {
    model: f"nav-pointnav2020-gibsonv2-light-{model}-evalval284-sjc-v1-seed0-20261022"
    for model in MODELS
}


def main():
    rows = {}
    for model in MODELS:
        path = ROOT / "results" / IDS[model] / "summary.json"
        row = json.loads(path.read_text())
        assert row["status"] == "complete" and row["completed_episodes"] == 284
        assert row["experiment_id"] == IDS[model]
        assert row["checkpoint_global_frames"] == 1_000_064
        assert row["validation_scenes"] == ["Cantwell", "Eastville", "Edgemere", "Scioto"]
        assert all(0 <= row["metrics"][metric] <= 1 for metric in ("success", "spl", "softspl"))
        rows[model] = row

    metrics = (("success", "SR"), ("spl", "SPL"), ("softspl", "SoftSPL"))
    fig, axis = plt.subplots(figsize=(8.2, 5.3), constrained_layout=True)
    centers = list(range(len(metrics)))
    width = 0.34
    for offset, model, label, color in ((-width / 2, "lstm2", "LSTM2", "#1976a3"),
                                        (width / 2, "gawf2", "GaWF2", "#dc6e29")):
        values = [100 * rows[model]["metrics"][key] for key, _ in metrics]
        bars = axis.bar([center + offset for center in centers], values,
                        width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value),
                          xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    axis.set_xticks(centers)
    axis.set_xticklabels([label for _, label in metrics])
    axis.set_ylabel("Validation score (%)")
    axis.set_ylim(0, max(5, max(100 * rows[model]["metrics"][key]
                            for model in MODELS for key, _ in metrics) * 1.22))
    axis.grid(axis="y", alpha=0.23)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    fig.suptitle("PointNav light · Gibson v2 validation · seed0", fontsize=13)
    axis.set_title("Four held-out scenes · 284 episodes per model · 1,000,064-frame checkpoints\n"
                   "Stochastic policy; 500-step limit; 0.36 m success radius", fontsize=9)
    files = {}
    for ext in ("png", "pdf"):
        buffer = BytesIO()
        fig.savefig(buffer, format=ext, dpi=180)
        files[f"pointnav_seed0_validation_sr_spl.{ext}"] = buffer.getvalue()
    plt.close(fig)
    receipt = {"collected_at": datetime.now().astimezone().isoformat(),
               "host": "sjc-remote", "dataset": "Gibson PointNav v2 validation",
               "scenes": ["Cantwell", "Eastville", "Edgemere", "Scioto"],
               "episodes_per_model": 284, "metric_scope": "validation episodes",
               "models": rows}
    files["validation_scores.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))


if __name__ == "__main__":
    main()
