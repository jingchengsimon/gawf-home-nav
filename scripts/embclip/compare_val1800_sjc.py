"""Compare eight completed EmbCLIP val runs on their shared episode IDs."""

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/embclip-val1800-seed01-20261022"
MODELS = ("gru1", "gawf1", "gru3", "gawf3")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return sum(values) / len(values)


def main():
    episode_files = sorted((ROOT / "data/embclip-robothor/robothor-objectnav/val/episodes").glob("*.json.gz"))
    expected = {item["id"] for path in episode_files
                for item in json.load(gzip.open(path, "rt"))}
    if len(episode_files) != 15 or len(expected) != 1800:
        raise RuntimeError("Expected 1,800 unique episodes from 15 val scenes")
    runs = []
    for model in MODELS:
        for seed in (0, 1):
            eid = f"nav-robothor-embclip-light-{model}-eval-val1800-sjc-v1-seed{seed}-20261022"
            manifest = json.loads((ROOT / "experiments/monitoring/jobs" / f"{eid}.json").read_text())
            result = ROOT / "results" / eid
            summary = json.loads((result / "summary.json").read_text())
            if (manifest["status"] not in ("complete", "partial")
                    or summary["status"] != manifest["status"]
                    or summary["checkpoint_sha256"] != manifest["source_checkpoint"]["sha256"]):
                raise RuntimeError(f"Unfinished or inconsistent evaluation: {eid}")
            metrics_path = ROOT / summary["metrics_path"]
            if sha256(metrics_path) != summary["metrics_sha256"]:
                raise RuntimeError(f"Metrics hash mismatch: {eid}")
            metrics = json.loads(metrics_path.read_text())
            if len(metrics) != 1:
                raise RuntimeError(f"Unexpected metrics shape: {eid}")
            tasks = {task["task_info"]["id"]: (float(task["success"]), float(task["spl"]))
                     for task in metrics[0]["tasks"]}
            if (len(tasks) != summary["episode_count"] or not set(tasks) <= expected
                    or sorted(expected - set(tasks)) != summary["missing_episode_ids"]):
                raise RuntimeError(f"Episode coverage mismatch: {eid}")
            runs.append(dict(experiment_id=eid, model=model, seed=seed,
                             status=summary["status"], episode_count=len(tasks),
                             success_rate=summary["success_rate"], spl=summary["spl"],
                             metrics_sha256=summary["metrics_sha256"], tasks=tasks))
    common = set.intersection(*(set(run["tasks"]) for run in runs))
    if not common:
        raise RuntimeError("No common validation episodes")
    for run in runs:
        tasks = run.pop("tasks")
        run["common_success_rate"] = mean([tasks[eid][0] for eid in common])
        run["common_spl"] = mean([tasks[eid][1] for eid in common])
    model_means = {}
    for model in MODELS:
        rows = [run for run in runs if run["model"] == model]
        model_means[model] = dict(
            success_rate=mean([row["common_success_rate"] for row in rows]),
            spl=mean([row["common_spl"] for row in rows]),
        )
    report = dict(split="robothor_objectnav_val_full_attempt",
                  expected_episode_count=1800, common_episode_count=len(common),
                  common_episode_ids=sorted(common), runs=runs, model_means=model_means,
                  paired_differences={
                      "gawf1_minus_gru1": {metric: model_means["gawf1"][metric] - model_means["gru1"][metric]
                                           for metric in ("success_rate", "spl")},
                      "gawf3_minus_gru3": {metric: model_means["gawf3"][metric] - model_means["gru3"][metric]
                                           for metric in ("success_rate", "spl")},
                  })
    OUT.mkdir(exist_ok=False)
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# EmbCLIP RoboTHOR ObjectNav validation, seed0/1", "",
             f"Official val contains 1,800 episodes; all eight runs share {len(common)} observed episode IDs.",
             "AllenAct evaluated with AI2-THOR 5.0.0, 500-step episodes and a fixed evaluation seed of 0.",
             "Partial runs are reported as such; common-set scores are comparable across all eight runs.", "",
             "| Model | Seed | Observed / 1,800 | Observed SR | Observed SPL | Common SR | Common SPL |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in runs:
        lines.append(f"| {run['model']} | {run['seed']} | {run['episode_count']} | "
                     f"{run['success_rate']:.4f} | {run['spl']:.4f} | "
                     f"{run['common_success_rate']:.4f} | {run['common_spl']:.4f} |")
    lines += ["", "| Model | Mean common SR | Mean common SPL |", "| --- | ---: | ---: |"]
    for model in MODELS:
        lines.append(f"| {model} | {model_means[model]['success_rate']:.4f} | {model_means[model]['spl']:.4f} |")
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Validated {len(runs)} runs on {len(common)} common episodes: {OUT}")


if __name__ == "__main__":
    main()
