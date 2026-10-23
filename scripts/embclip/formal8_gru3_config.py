from scripts.embclip.formal_gru3_config import FormalGRU3Config
from scripts.embclip.formal8_config import Formal8MachineMixin


class Formal8GRU3Config(Formal8MachineMixin, FormalGRU3Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-GRU3-331-200M-8GPU"
