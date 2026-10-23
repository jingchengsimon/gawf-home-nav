"""Five-million-frame GRU1 EmbCLIP run."""

from scripts.embclip.light_gru1_config import LightGRU1Config


class Light5MGRU1Config(LightGRU1Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-GRU1-512-5M"
