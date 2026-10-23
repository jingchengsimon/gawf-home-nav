from scripts.embclip.formal_config import FormalEmbCLIPConfig


class FormalGRU1Config(FormalEmbCLIPConfig):
    MODEL = "gru1"
    HIDDEN_SIZE = 512

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-GRU1-512-200M"
