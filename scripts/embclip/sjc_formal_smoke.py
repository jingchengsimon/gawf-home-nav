"""Use SJC's two GPUs for a bounded comparison-model smoke only."""

from allenact.base_abstractions.experiment_config import MachineParams
from projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig as OfficialConfig,
)


class SJCFormalSmokeMixin:
    def machine_params(self, mode="train", **kwargs):
        if mode == "valid":
            return MachineParams(nprocesses=0, devices=[-1])
        return OfficialConfig.machine_params(self, mode=mode, **kwargs)

    def training_pipeline(self, **kwargs):
        pipeline = super().training_pipeline(**kwargs)
        pipeline.pipeline_stages[0].max_stage_steps = 60 * 128
        return pipeline
