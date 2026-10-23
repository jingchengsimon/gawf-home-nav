"""Run an unchanged 200M EmbCLIP model on one SJC GPU."""

from allenact.base_abstractions.experiment_config import MachineParams
from projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig,
)


class SJCSingleGPUFormalMixin:
    def machine_params(self, mode="train", **kwargs):
        if mode == "valid":
            return MachineParams(nprocesses=0, devices=[-1])
        if mode == "train":
            assert self.num_train_processes == 60
            assert len(self.train_gpu_ids) == 1
        return ObjectNavRoboThorClipRGBPPOExperimentConfig.machine_params(
            self, mode=mode, **kwargs
        )
