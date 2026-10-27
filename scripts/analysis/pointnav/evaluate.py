"""Evaluate a completed RGB-only PointNav seed on the Gibson v2 val episodes."""

import argparse
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# Forkserver children must install the pinned Habitat compatibility layer first.
multiprocessing.set_forkserver_preload(["scripts.training.pointnav.runtime"])
from scripts.training.pointnav.runtime import OBSERVATIONS


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    args = parser.parse_args()
    manifest_path = ROOT / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["experiment_id"] == args.experiment_id
    assert manifest["host"] == "sjc-remote" and manifest["mode"] == "checkpoint_eval"
    assert manifest["protocol_id"] == "pointnav2020-gibsonv2-light-rgb-fixedgoal-resnet18-v1"
    assert manifest["model"] in ("lstm2", "gawf2") and manifest["seed"] == 0
    assert manifest["expected_episodes"] == 284
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert sha256(Path(__file__)) == manifest["runner_sha256"]
    for relative, expected in manifest["source_files"].items():
        assert sha256(ROOT / relative) == expected, f"Source mismatch: {relative}"

    scenes = ("Cantwell", "Eastville", "Edgemere", "Scioto")
    assert manifest["datasets"]["validation_scenes"] == list(scenes)
    result = ROOT / "results" / args.experiment_id
    assert result.is_dir() and result.resolve().parent == (ROOT / "results").resolve()
    assert not (result / "summary.json").exists()
    source_id = manifest["source_experiment_id"]
    checkpoint = ROOT / "results" / source_id / "checkpoints/interrupted.pth"
    source_summary = json.loads((checkpoint.parent.parent / "summary.json").read_text())
    exit_receipt = json.loads((checkpoint.parent.parent / "exit.attempt001.json").read_text())
    assert source_summary["status"] == "complete"
    assert source_summary["global_frames"] == 1_000_064
    assert exit_receipt["return_code"] == 0 and exit_receipt["complete"]
    assert exit_receipt["saved_global_frames"] == 1_000_064
    digest = sha256(checkpoint)
    assert digest == source_summary["checkpoint_sha256"] == manifest["checkpoint_sha256"]

    import numpy as np
    import torch
    from habitat_baselines.config.default import get_config
    from habitat_baselines.rl.ddppo.algo import ddp_utils
    from habitat_baselines.rl.ddppo.algo.ddppo_trainer import DDPPOTrainer
    from scripts.training.pointnav.trainer import configure_formal_policy

    torch.set_num_threads(1)
    state = torch.load(checkpoint, map_location="cpu")
    assert state["formal_meta"]["model"] == manifest["model"]
    assert state["formal_meta"]["seed"] == 0
    assert state["requeue_stats"]["count_steps"] == 1_000_064
    assert {"state_dict", "optim_state", "lr_sched_state", "config"} <= set(state)
    del state

    config = get_config(str(ROOT / "configs/pointnav2020/light_rgb/ddppo.yaml"), [
        "BASE_TASK_CONFIG_PATH", str(ROOT / "configs/pointnav2020/light_rgb/task.yaml"),
        "TASK_CONFIG.DATASET.DATA_PATH", str(ROOT / "data/datasets/pointnav/gibson/v2/{split}/{split}.json.gz"),
        "TASK_CONFIG.DATASET.SCENES_DIR", str(ROOT / "data/scene_datasets"),
        "TASK_CONFIG.DATASET.SPLIT", "val",
        "TASK_CONFIG.SEED", "0",
        "ENV_NAME", "CheckedRGBPointNav2020",
        "CHECKPOINT_FOLDER", str(result / "checkpoints"),
    ])
    config.defrost()
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = list(scenes)
    config.TEST_EPISODE_COUNT = 284
    config.EVAL.SPLIT = "val"
    config.EVAL.USE_CKPT_CONFIG = False
    config.EVAL_CKPT_PATH_DIR = str(checkpoint)
    config.TENSORBOARD_DIR = str(result / "tensorboard")
    config.LOG_FILE = str(result / "eval.log")
    config.VIDEO_OPTION = []
    config.freeze()
    assert set(OBSERVATIONS) == {"rgb", "pointgoal"}
    assert list(config.SENSORS) == ["RGB_SENSOR"]
    assert list(config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS) == ["RGB_SENSOR"]
    assert list(config.TASK_CONFIG.TASK.SENSORS) == ["POINTGOAL_SENSOR"]
    assert config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID == "pointgoal"
    assert config.NUM_PROCESSES == 2
    assert config.RL.DDPPO.backbone == "resnet18"
    assert config.RL.DDPPO.num_recurrent_layers == 2
    assert config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS == 500
    assert config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE == 0.36

    ddp_utils.INTERRUPTED_STATE_FILE = str(result / "unused-interrupted.pth")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    trainer = DDPPOTrainer(config)
    configure_formal_policy(manifest["model"], trainer)
    original_setup = trainer._setup_actor_critic_agent

    def checked_setup(ppo_cfg):
        original_setup(ppo_cfg)

        def check_observations(_module, inputs):
            assert set(inputs[0]) == OBSERVATIONS

        trainer.actor_critic.net.register_forward_pre_hook(check_observations)

    trainer._setup_actor_critic_agent = checked_setup
    completed = []
    original_metrics = trainer._extract_scalars_from_info

    def measured_metrics(info):
        metrics = original_metrics(info)
        assert "success" in metrics and "spl" in metrics
        assert all(math.isfinite(float(value)) for value in metrics.values())
        completed.append(metrics)
        if len(completed) % 10 == 0 or len(completed) == 284:
            write_json(result / "progress.json", dict(status="running",
                completed_episodes=len(completed), expected_episodes=284,
                updated_at=datetime.now().astimezone().isoformat()))
        return metrics

    trainer._extract_scalars_from_info = measured_metrics
    manifest["status"] = "running"
    manifest["units"][0]["status"] = "running"
    write_json(manifest_path, manifest)
    write_json(result / "progress.json", dict(status="running", completed_episodes=0,
        expected_episodes=284, updated_at=datetime.now().astimezone().isoformat()))
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/training/monitor.py"), "--rebuild"], check=True)
    trainer.eval()
    assert len(completed) == 284, f"Expected 284 episodes, got {len(completed)}"
    metric_names = set(completed[0])
    assert all(set(row) == metric_names for row in completed)
    averages = {key: sum(float(row[key]) for row in completed) / len(completed)
                for key in sorted(metric_names)}
    summary = dict(status="complete", experiment_id=args.experiment_id,
        source_experiment_id=source_id, checkpoint_sha256=digest,
        checkpoint_global_frames=1_000_064, completed_episodes=284,
        validation_scenes=list(scenes), metrics=averages,
        completed_at=datetime.now().astimezone().isoformat())
    write_json(result / "summary.json", summary)
    write_json(result / "progress.json", dict(status="complete", completed_episodes=284,
        expected_episodes=284, updated_at=summary["completed_at"]))
    manifest["status"] = "complete"
    manifest["units"][0]["status"] = "complete"
    manifest["units"][0]["metrics"] = {"completed_episodes": 284, **averages}
    manifest["completion_evidence"] = {"episode_count": 284,
        "summary_path": str(result / "summary.json"), "checkpoint_sha256": digest}
    write_json(manifest_path, manifest)
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/training/monitor.py"), "--rebuild"], check=True)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
