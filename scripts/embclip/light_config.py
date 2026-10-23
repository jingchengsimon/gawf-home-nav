"""Short RoboTHOR EmbCLIP comparison with the official frozen RN50 encoder."""

import gym

from allenact.base_abstractions.experiment_config import MachineParams
from allenact_plugins.navigation_plugin.objectnav.models import ResnetTensorNavActorCritic
from projects.objectnav_baselines.mixins import ObjectNavPPOMixin
from projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig,
)

from scripts.embclip.sjc_config import DATA


TARGET_STEPS = 1_000_000
GLOBAL_ENVIRONMENTS = 16
CHECKPOINT_INTERVAL = 100_000


class LightEmbCLIPConfig(ObjectNavRoboThorClipRGBPPOExperimentConfig):
    MODEL = "gru1"
    HIDDEN_SIZE = 512
    RNN_TYPE = "GRU"
    NUM_LAYERS = 1

    TRAIN_DATASET_DIR = str(DATA / "train")
    VAL_DATASET_DIR = str(DATA / "val")
    TEST_DATASET_DIR = None

    def __init__(self, target_steps=TARGET_STEPS, checkpoint_interval=CHECKPOINT_INTERVAL, **kwargs):
        self.target_steps = int(target_steps)
        self.checkpoint_interval = int(checkpoint_interval)
        if self.target_steps <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("target_steps and checkpoint_interval must be positive")
        super().__init__(**kwargs)

    def create_model(self, **kwargs):
        return ResnetTensorNavActorCritic(
            action_space=gym.spaces.Discrete(self.ACTION_SPACE.n),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=self.SENSORS[1].uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet",
            depth_resnet_preprocessor_uuid=None,
            hidden_size=self.HIDDEN_SIZE,
            num_rnn_layers=self.NUM_LAYERS,
            rnn_type=self.RNN_TYPE,
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
