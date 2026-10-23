from scripts.embclip.formal_gawf3_config import FormalGaWF3Config
from scripts.embclip.sjc_formal_smoke import SJCFormalSmokeMixin


class SJCGaWF3SmokeConfig(SJCFormalSmokeMixin, FormalGaWF3Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-GaWF3-653-SJC-Smoke"
