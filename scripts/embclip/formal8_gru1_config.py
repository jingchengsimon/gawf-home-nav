from scripts.embclip.formal_gru1_config import FormalGRU1Config
from scripts.embclip.formal8_config import Formal8MachineMixin


class Formal8GRU1Config(Formal8MachineMixin, FormalGRU1Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-GRU1-512-200M-8GPU"
