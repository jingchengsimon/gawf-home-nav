from scripts.embclip.formal_gru3_config import FormalGRU3Config
from scripts.embclip.sjc_single_gpu_formal import SJCSingleGPUFormalMixin


class SJCGRU3FormalConfig(SJCSingleGPUFormalMixin, FormalGRU3Config):
    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-GRU3-331-200M-SJC1GPU"
