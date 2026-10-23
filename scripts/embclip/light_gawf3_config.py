"""Three-layer GaWF paired with the light GRU3 EmbCLIP baseline."""

import gym

from scripts.embclip.formal_config import ThreeLayerGaWFActorCritic
from scripts.embclip.light_config import LightEmbCLIPConfig


class LightGaWF3Config(LightEmbCLIPConfig):
    MODEL = "gawf3"
    HIDDEN_SIZE = 653
    NUM_LAYERS = 3

    def create_model(self, **kwargs):
        return ThreeLayerGaWFActorCritic(
            action_space=gym.spaces.Discrete(self.ACTION_SPACE.n),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=self.SENSORS[1].uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet",
            depth_resnet_preprocessor_uuid=None,
            hidden_size=self.HIDDEN_SIZE,
            goal_dims=32,
        )

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-GaWF3-653-1M"
