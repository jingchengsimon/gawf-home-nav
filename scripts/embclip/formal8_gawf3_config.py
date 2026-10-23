from scripts.embclip.formal_gawf3_config import FormalGaWF3Config
from scripts.embclip.formal8_config import Formal8MachineMixin


class Formal8GaWF3Config(Formal8MachineMixin, FormalGaWF3Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-GaWF3-653-200M-8GPU"
