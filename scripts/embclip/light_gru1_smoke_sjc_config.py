from scripts.embclip.light_gru1_config import LightGRU1Config
from scripts.embclip.light_smoke_sjc import LightSmokeMixin


class LightGRU1SmokeSJCConfig(LightSmokeMixin, LightGRU1Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-GRU1-512-SJC-Smoke"
