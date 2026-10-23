from scripts.embclip.formal_config import FormalEmbCLIPConfig


class FormalGaWF3Config(FormalEmbCLIPConfig):
    MODEL = "gawf3"
    HIDDEN_SIZE = 653

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-GaWF3-653-200M"
