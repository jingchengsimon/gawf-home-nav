"""Bounded SJC throughput smoke for the light Gibson v2 PointNav task."""

import json
from pathlib import Path
import sys

import train_pointnav2020 as base
from pointnav2020_runtime import ROOT
from train_pointnav2020_light import light_config


def benchmark_config(experiment_id, updates=4, split="train", seed=0,
                     checkpoint_interval=1, log_interval=1):
    manifest = json.loads((ROOT / "experiments/monitoring/jobs" /
                           (experiment_id + ".json")).read_text())
    assert manifest["experiment_id"] == experiment_id
    assert manifest["host"] == "sjc-remote" and manifest["mode"] == "train_smoke"
    assert manifest["model"] == "lstm2" and seed == manifest["seed"] == 0
    environments = manifest["budget"]["environments_per_gpu"]
    rollout = manifest["budget"]["rollout_steps_per_environment"]
    assert (environments, rollout, updates) in {(2, 64, 16), (4, 64, 8), (2, 128, 8)}
    config = light_config(experiment_id, updates=updates, split=split, seed=seed,
                          checkpoint_interval=1000, log_interval=1)
    config.defrost()
    config.NUM_PROCESSES = environments
    config.RL.PPO.num_steps = rollout
    config.freeze()
    return config


base.load_config = benchmark_config

if __name__ == "__main__":
    assert "--formal" not in sys.argv
    base.main()
