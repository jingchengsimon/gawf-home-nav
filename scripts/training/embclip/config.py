"""Five-million-frame RoboTHOR EmbCLIP comparison with the official frozen RN50 encoder."""

import gym

from allenact.base_abstractions.experiment_config import MachineParams
from allenact_plugins.navigation_plugin.objectnav.models import ResnetTensorNavActorCritic
from projects.objectnav_baselines.mixins import ObjectNavPPOMixin
from projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig,
)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/embclip-robothor/robothor-objectnav"


TARGET_STEPS = 5_000_000
GLOBAL_ENVIRONMENTS = 16
CHECKPOINT_INTERVAL = 100_000


class EmbCLIPConfig(ObjectNavRoboThorClipRGBPPOExperimentConfig):
    MODEL = "gru1"
    HIDDEN_SIZE = 512
    RNN_TYPE = "GRU"
    NUM_LAYERS = 1

    TRAIN_DATASET_DIR = str(DATA / "train")
    VAL_DATASET_DIR = str(DATA / "val")
    TEST_DATASET_DIR = None

    def __init__(self, model="gru1", official=False, target_steps=TARGET_STEPS,
                 checkpoint_interval=CHECKPOINT_INTERVAL, **kwargs):
        if model not in {"gru1", "gawf1"} or (official and model != "gru1"):
            raise ValueError("Expected GRU1/GaWF1, or the released official GRU1")
        self.MODEL = model
        self.HIDDEN_SIZE = 512 if model == "gru1" else 1160
        if official:
            self._configure_official_cache()
        self.target_steps = int(target_steps)
        self.checkpoint_interval = int(checkpoint_interval)
        if self.target_steps <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("target_steps and checkpoint_interval must be positive")
        super().__init__(**kwargs)

    def create_model(self, **kwargs):
        from scripts.training.embclip.models import SingleLayerGaWFActorCritic

        actor = SingleLayerGaWFActorCritic if self.MODEL == "gawf1" else ResnetTensorNavActorCritic
        recurrent = {} if self.MODEL == "gawf1" else {"num_rnn_layers": 1, "rnn_type": "GRU"}
        return actor(
            action_space=gym.spaces.Discrete(self.ACTION_SPACE.n),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=self.SENSORS[1].uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet",
            depth_resnet_preprocessor_uuid=None,
            hidden_size=self.HIDDEN_SIZE,
            **recurrent,
            goal_dims=32,
        )

    def training_pipeline(self, **kwargs):
        pipeline = ObjectNavPPOMixin.training_pipeline(
            auxiliary_uuids=[],
            multiple_beliefs=False,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            save_interval=self.checkpoint_interval,
        )
        pipeline.pipeline_stages[0].max_stage_steps = self.target_steps
        return pipeline

    def machine_params(self, mode="train", **kwargs):
        if mode == "valid":
            return MachineParams(nprocesses=0, devices=[-1])
        if mode == "train":
            assert self.num_train_processes == GLOBAL_ENVIRONMENTS
            assert len(self.train_gpu_ids) == 1
        return super().machine_params(mode=mode, **kwargs)

    @staticmethod
    def _configure_official_cache():
        import clip
        from ai2thor.controller import Controller

        Controller.base_dir = property(lambda self: str(ROOT / "assets/embclip-robothor/ai2thor"))
        if not getattr(clip.load, "_official_val_cache", False):
            original = clip.load

            def cached_load(*args, **kwargs):
                kwargs.setdefault("download_root", str(ROOT / "assets/embclip-robothor/home/.cache/clip"))
                return original(*args, **kwargs)

            cached_load._official_val_cache = True
            clip.load = cached_load

    def tag(self):
        label = "GRU1-512" if self.MODEL == "gru1" else "GaWF1-1160"
        return f"ObjectNav-RoboTHOR-EmbCLIP-Light-{label}-5M"
