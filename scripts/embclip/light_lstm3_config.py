from scripts.embclip.light_config import LightEmbCLIPConfig


class LightLSTM3Config(LightEmbCLIPConfig):
    MODEL = "lstm3"
    HIDDEN_SIZE = 272
    NUM_LAYERS = 3
    RNN_TYPE = "LSTM"

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-EmbCLIP-Light-LSTM3-272-1M"
