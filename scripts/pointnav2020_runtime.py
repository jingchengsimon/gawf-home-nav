"""Load the pinned 2020 stack and enforce its observation contract."""

import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "vendor/habitat-lab-2020"
sys.path.insert(0, str(LEGACY))
from pointnav2020_compat import install

install(ROOT)
# Legacy config defaults resolve relative to this vendor root.
os.chdir(LEGACY)

import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv
from habitat_baselines.config.default import get_config

OBSERVATIONS = {"rgb", "depth", "pointgoal"}


@baseline_registry.register_env(name="CheckedPointNav2020")
class CheckedPointNav2020(NavRLEnv):
    """Assertions only; original environment actions/rewards/termination unchanged."""

    def reset(self):
        observations = super().reset()
        assert set(observations) == OBSERVATIONS
        self._checked_goal = observations["pointgoal"].copy()
        return observations

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)
        assert set(observations) == OBSERVATIONS
        assert np.array_equal(observations["pointgoal"], self._checked_goal)
        assert np.isfinite(reward)
        return observations, reward, done, info


def run_paths(experiment_id):
    if not re.fullmatch(r"nav-[a-z0-9]+(?:-[a-z0-9]+)+", experiment_id):
        raise ValueError("Complete experiment ID required")
    return ROOT / "results" / experiment_id, ROOT / "logs" / experiment_id


def load_config(experiment_id, updates=4, split="train", seed=0,
                checkpoint_interval=1, log_interval=1):
    results, logs = run_paths(experiment_id)
    config = get_config(str(ROOT / "configs/pointnav2020/ddppo.yaml"), [
        "BASE_TASK_CONFIG_PATH", str(ROOT / "configs/pointnav2020/task.yaml"),
        "TASK_CONFIG.DATASET.DATA_PATH", str(ROOT / "data/datasets/pointnav/gibson/v2/{split}/{split}.json.gz"),
        "TASK_CONFIG.DATASET.SCENES_DIR", str(ROOT / "data/scene_datasets"),
        "TASK_CONFIG.DATASET.SPLIT", split,
        "TASK_CONFIG.SEED", str(seed),
        "ENV_NAME", "CheckedPointNav2020",
        "NUM_UPDATES", str(updates),
        "LOG_INTERVAL", str(log_interval),
        "CHECKPOINT_INTERVAL", str(checkpoint_interval),
        "CHECKPOINT_FOLDER", str(results / "checkpoints"),
        "EVAL_CKPT_PATH_DIR", str(results / "checkpoints"),
        "TENSORBOARD_DIR", str(logs / "tensorboard"),
        "VIDEO_DIR", str(results / "video"),
        "LOG_FILE", str(logs / "train.log"),
    ])
    assert config.TASK_CONFIG.TASK.SENSORS == ["POINTGOAL_SENSOR"]
    assert set(config.SENSORS) == {"RGB_SENSOR", "DEPTH_SENSOR"}
    return config
