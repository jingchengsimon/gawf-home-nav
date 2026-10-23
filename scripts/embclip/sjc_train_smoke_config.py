"""One-update SJC smoke of the frozen EmbCLIP training pipeline."""
from allenact.base_abstractions.experiment_config import MachineParams

from scripts.embclip.sjc_config import SJCEmbCLIPConfig


class SJCEmbCLIPTrainSmokeConfig(SJCEmbCLIPConfig):
    def training_pipeline(self, **kwargs):
        pipeline = super().training_pipeline(**kwargs)
        # End cleanly after at least one 60 x 128 rollout. The inherited
        # optimizer, PPO settings, and 300M-step LR schedule remain intact.
        pipeline.pipeline_stages[0].max_stage_steps = 60 * 128
        return pipeline

    def machine_params(self, mode="train", **kwargs):
        if mode == "valid":
            # Checkpoint evaluation is tracked in its separate SJC experiment.
            return MachineParams(nprocesses=0, devices=[-1])
        return super().machine_params(mode=mode, **kwargs)
