"""Five-million-frame GaWF3 EmbCLIP run."""

from scripts.embclip.light_gawf3_config import LightGaWF3Config


class Light5MGaWF3Config(LightGaWF3Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-GaWF3-653-5M"
