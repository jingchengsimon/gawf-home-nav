from scripts.embclip.light_config import LightEmbCLIPConfig


class LightGRU3Config(LightEmbCLIPConfig):
    MODEL = "gru3"
    HIDDEN_SIZE = 331
    NUM_LAYERS = 3

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-GRU3-331-1M"
