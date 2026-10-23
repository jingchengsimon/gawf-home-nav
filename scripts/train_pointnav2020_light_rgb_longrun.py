"""RGB-only light PointNav training with the existing complete-state DD-PPO loop."""

import sys

import train_pointnav2020 as base
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv
from habitat_baselines.config.default import get_config
from pointnav2020_runtime import ROOT

OBSERVATIONS = {"rgb", "pointgoal"}
TRAIN_SCENES = {"Convoy", "Eagerville", "Haxtun", "Nuevo", "Reyno", "Sawpit", "Seward", "Sumas"}


@baseline_registry.register_env(name="CheckedRGBPointNav2020")
class CheckedRGBPointNav2020(NavRLEnv):
    def reset(self):
        observations = super().reset()
        assert set(observations) == OBSERVATIONS
        self._initial_goal = observations["pointgoal"].copy()
        return observations

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)
        assert set(observations) == OBSERVATIONS
        assert (observations["pointgoal"] == self._initial_goal).all()
        return observations, reward, done, info


def rgb_config(experiment_id, updates=4, split="train", seed=0,
               checkpoint_interval=1, log_interval=1):
    assert updates == 50000 and split == "train"
    results, logs = base.run_paths(experiment_id)
    config = get_config(str(ROOT / "configs/pointnav2020/light_rgb/ddppo.yaml"), [
        "BASE_TASK_CONFIG_PATH", str(ROOT / "configs/pointnav2020/light_rgb/task.yaml"),
        "TASK_CONFIG.DATASET.DATA_PATH", str(ROOT / "data/datasets/pointnav/gibson/v2/{split}/{split}.json.gz"),
        "TASK_CONFIG.DATASET.SCENES_DIR", str(ROOT / "data/scene_datasets"),
        "TASK_CONFIG.DATASET.SPLIT", split,
        "TASK_CONFIG.SEED", str(seed),
        "ENV_NAME", "CheckedRGBPointNav2020",
        "NUM_UPDATES", "1000000",
        "LOG_INTERVAL", str(log_interval),
        "CHECKPOINT_INTERVAL", str(checkpoint_interval),
        "CHECKPOINT_FOLDER", str(results / "checkpoints"),
        "EVAL_CKPT_PATH_DIR", str(results / "checkpoints"),
        "TENSORBOARD_DIR", str(logs / "tensorboard"),
        "VIDEO_DIR", str(results / "video"),
        "LOG_FILE", str(logs / "train.log"),
    ])
    assert list(config.SENSORS) == ["RGB_SENSOR"]
    assert list(config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS) == ["RGB_SENSOR"]
    assert list(config.TASK_CONFIG.TASK.SENSORS) == ["POINTGOAL_SENSOR"]
    assert config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID == "pointgoal"
    assert set(config.TASK_CONFIG.DATASET.CONTENT_SCENES) == TRAIN_SCENES
    assert config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH == 256
    assert config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT == 144
    assert config.RL.DDPPO.backbone == "resnet18"
    assert not config.RL.DDPPO.pretrained and not config.RL.DDPPO.pretrained_encoder
    assert config.NUM_PROCESSES == 2 and config.RL.PPO.num_steps == 64
    assert not config.RL.PPO.use_linear_lr_decay and not config.RL.PPO.use_linear_clip_decay
    return config


base.OBSERVATIONS = OBSERVATIONS
base.load_config = rgb_config


if __name__ == "__main__":
    assert "--formal" in sys.argv
    base.main()
