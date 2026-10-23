from scripts.embclip.light_lstm3_config import LightLSTM3Config
from scripts.embclip.light_smoke_sjc import LightSmokeMixin


class LightLSTM3SmokeSJCConfig(LightSmokeMixin, LightLSTM3Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-LSTM3-272-SJC-Smoke"
