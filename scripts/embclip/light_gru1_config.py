from scripts.embclip.light_config import LightEmbCLIPConfig


class LightGRU1Config(LightEmbCLIPConfig):
    MODEL = "gru1"
    HIDDEN_SIZE = 512

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-GRU1-512-1M"
