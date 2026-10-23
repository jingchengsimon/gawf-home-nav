"""Use the verified 2020 formal trainer with a separate light task/config."""

import train_pointnav2020 as base
from pointnav2020_runtime import ROOT, OBSERVATIONS
from habitat_baselines.config.default import get_config


def light_config(experiment_id, updates=4, split="train", seed=0,
                 checkpoint_interval=1, log_interval=1):
    results, logs = base.run_paths(experiment_id)
    task = ROOT / "configs/pointnav2020/light/task.yaml"
    config = get_config(str(ROOT / "configs/pointnav2020/light/ddppo.yaml"), [
        "BASE_TASK_CONFIG_PATH", str(task),
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
    assert set(config.SENSORS) == {"RGB_SENSOR", "DEPTH_SENSOR"}
    assert config.TASK_CONFIG.TASK.SENSORS == ["POINTGOAL_SENSOR"]
    assert set(config.TASK_CONFIG.DATASET.CONTENT_SCENES) == {
        "Convoy", "Eagerville", "Haxtun", "Nuevo", "Reyno", "Sawpit", "Seward", "Sumas"
    }
    assert config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH == 256
    assert config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT == 144
    assert config.RL.DDPPO.backbone == "resnet18"
    assert config.NUM_PROCESSES == 2
    assert set(OBSERVATIONS) == {"rgb", "depth", "pointgoal"}
    return config


base.load_config = light_config

if __name__ == "__main__":
    base.main()
