"""Continue the SJC EmbCLIP smoke checkpoint for one further rollout."""
from scripts.embclip.sjc_train_smoke_config import SJCEmbCLIPTrainSmokeConfig


class SJCEmbCLIPResumeSmokeConfig(SJCEmbCLIPTrainSmokeConfig):
    def training_pipeline(self, **kwargs):
        pipeline = super().training_pipeline(**kwargs)
        pipeline.pipeline_stages[0].max_stage_steps = 2 * 60 * 128
        return pipeline
