from scripts.embclip.formal_config import FormalEmbCLIPConfig


class FormalGRU3Config(FormalEmbCLIPConfig):
    MODEL = "gru3"
    HIDDEN_SIZE = 331

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-GRU3-331-200M"
