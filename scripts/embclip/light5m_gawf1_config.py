"""Five-million-frame GaWF1 EmbCLIP run."""

from scripts.embclip.light_gawf1_config import LightGaWF1Config


class Light5MGaWF1Config(LightGaWF1Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-GaWF1-1160-5M"
