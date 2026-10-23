from scripts.embclip.formal_gru3_config import FormalGRU3Config
from scripts.embclip.sjc_formal_smoke import SJCFormalSmokeMixin


class SJCGRU3SmokeConfig(SJCFormalSmokeMixin, FormalGRU3Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-GRU3-331-SJC-Smoke"
