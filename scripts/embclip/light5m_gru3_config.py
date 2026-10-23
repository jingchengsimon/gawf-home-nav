"""Five-million-frame GRU3 EmbCLIP run."""

from scripts.embclip.light_gru3_config import LightGRU3Config


class Light5MGRU3Config(LightGRU3Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-GRU3-331-5M"
