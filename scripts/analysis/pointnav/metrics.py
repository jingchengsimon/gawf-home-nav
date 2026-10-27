"""Verify current PointNav summaries and emit their structured numeric comparison."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/pointnav-rgb-seed0-validation-20261023"
MODELS = ("lstm2", "gawf2")
SCENES = ["Cantwell", "Eastville", "Edgemere", "Scioto"]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def collect_validation():
    data = {}
    for model in MODELS:
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

    snapshot = dict(created_at=datetime.now().astimezone().isoformat(),
                    protocol="pointnav2020-gibsonv2-light-rgb-fixedgoal-resnet18-v1",
                    dataset="Gibson PointNav v2 val", scenes=SCENES,
                    episodes_per_model=284, plot="mean bar plus seed scatter",
                    models=data)
    return snapshot



def main():
    global ROOT, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    OUT = args.source or ROOT / "results/pointnav-rgb-seed0-validation-20261023"
    print(json.dumps(collect_validation(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
