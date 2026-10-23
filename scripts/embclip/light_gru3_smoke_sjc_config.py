from scripts.embclip.light_gru3_config import LightGRU3Config
from scripts.embclip.light_smoke_sjc import LightSmokeMixin


class LightGRU3SmokeSJCConfig(LightSmokeMixin, LightGRU3Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-GRU3-331-SJC-Smoke"
