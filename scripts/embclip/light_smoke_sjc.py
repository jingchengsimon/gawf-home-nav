"""One rollout on SJC for each lightweight EmbCLIP model."""


class LightSmokeMixin:
    def training_pipeline(self, **kwargs):
        pipeline = super().training_pipeline(**kwargs)
        pipeline.pipeline_stages[0].max_stage_steps = 16 * 128
        pipeline.save_interval = 16 * 128
        return pipeline
