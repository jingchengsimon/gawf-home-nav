"""Use the light PointNav task with an update ceiling above its frame target."""

import sys

import train_pointnav2020 as base
from train_pointnav2020_light import light_config


def longrun_config(experiment_id, updates=4, split="train", seed=0,
                   checkpoint_interval=1, log_interval=1):
    assert updates == 50000, "The formal entrypoint must supply its expected ceiling"
    config = light_config(experiment_id, updates=1_000_000, split=split, seed=seed,
                          checkpoint_interval=checkpoint_interval,
                          log_interval=log_interval)
    assert not config.RL.PPO.use_linear_lr_decay
    assert not config.RL.PPO.use_linear_clip_decay
    return config


base.load_config = longrun_config

if __name__ == "__main__":
    assert "--formal" in sys.argv, "This entrypoint is for measured-frame training"
    base.main()
