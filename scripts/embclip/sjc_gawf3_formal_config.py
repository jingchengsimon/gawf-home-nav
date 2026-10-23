from scripts.embclip.formal_gawf3_config import FormalGaWF3Config
from scripts.embclip.sjc_single_gpu_formal import SJCSingleGPUFormalMixin


class SJCGaWF3FormalConfig(SJCSingleGPUFormalMixin, FormalGaWF3Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-GaWF3-653-200M-SJC1GPU"
