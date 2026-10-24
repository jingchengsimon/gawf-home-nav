"""Official released GRU checkpoint, using the light runs' Val protocol."""
import os
from pathlib import Path

import clip
from ai2thor.controller import Controller
from scripts.embclip.light5m_gru1_config import Light5MGRU1Config


_root = Path(os.environ['GAWF_HOME_NAV_ROOT']).resolve()
Controller.base_dir = property(lambda self: str(_root / 'assets/embclip-robothor/ai2thor'))
if not getattr(clip.load, '_official_val_cache', False):
    _load = clip.load

    def _cached_load(*args, **kwargs):
        kwargs.setdefault('download_root', str(_root / 'assets/embclip-robothor/home/.cache/clip'))
        return _load(*args, **kwargs)

    _cached_load._official_val_cache = True
    clip.load = _cached_load


class OfficialReleasedEmbCLIPValConfig(Light5MGRU1Config):
    @classmethod
    def tag(cls):
        return 'ObjectNav-RoboTHOR-EmbCLIP-Official-Released-GRU1-130M-Val'
