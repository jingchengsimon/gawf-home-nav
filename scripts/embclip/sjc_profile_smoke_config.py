"""Bounded SJC EmbCLIP profile with the official 300M-step LR schedule."""
from scripts.embclip.sjc_resume_smoke_config import SJCEmbCLIPResumeSmokeConfig


class SJCEmbCLIPProfileSmokeConfig(SJCEmbCLIPResumeSmokeConfig):
    def training_pipeline(self, **kwargs):
        pipeline = super().training_pipeline(**kwargs)
        pipeline.pipeline_stages[0].max_stage_steps = 10 * 60 * 128
        return pipeline
