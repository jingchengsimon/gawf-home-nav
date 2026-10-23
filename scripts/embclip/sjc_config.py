"""SJC path adapter for the frozen official RoboTHOR CLIP RN50 + GRU config."""
import os
from pathlib import Path

from projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/embclip-robothor/robothor-objectnav"


class SJCEmbCLIPConfig(ObjectNavRoboThorClipRGBPPOExperimentConfig):
    TRAIN_DATASET_DIR = str(DATA / "train")
    VAL_DATASET_DIR = os.environ.get("EMBCLIP_VAL_DATASET_DIR", str(DATA / "val"))
    TEST_DATASET_DIR = None
