"""Load the unchanged 2020 policy without importing obsolete training entrypoints."""

import importlib
from pathlib import Path
import sys
import types


def load_baseline_namespace(root: Path, vendor="habitat-lab-2020"):
    source = root / "vendor" / vendor / "habitat_baselines"
    if "habitat_baselines" in sys.modules:
        raise RuntimeError("Load legacy policy before importing modern habitat_baselines")
    for suffix in ("", ".common", ".utils", ".rl", ".rl.models", ".rl.ppo", ".rl.ddppo", ".rl.ddppo.policy"):
        name = "habitat_baselines" + suffix
        module = types.ModuleType(name)
        module.__path__ = [str(source.joinpath(*suffix.strip(".").split("."))) if suffix else str(source)]
        sys.modules[name] = module
    policy = importlib.import_module("habitat_baselines.rl.ppo.policy")
    sys.modules["habitat_baselines.rl.ppo"].Net = policy.Net
    sys.modules["habitat_baselines.rl.ppo"].Policy = policy.Policy


def load_2020_policy_class(root: Path):
    load_baseline_namespace(root)
    return importlib.import_module("habitat_baselines.rl.ddppo.policy.resnet_policy").PointNavResNetPolicy


def load_zer_policy_class(root: Path):
    import habitat
    import habitat.config
    import habitat.core.simulator
    from omegaconf import DictConfig

    # Import-only aliases for old annotations and the renamed Gym module.
    habitat.Config = DictConfig
    habitat.config.Config = DictConfig
    habitat.core.simulator.Config = DictConfig
    sys.modules["gym.spaces.dict_space"] = importlib.import_module("gym.spaces.dict")
    load_baseline_namespace(root, "habitat-lab-zer")
    source = root / "vendor/zero_experience_required"
    sys.path.insert(0, str(source))
    rl = types.ModuleType("rl")
    rl.__path__ = [str(source / "rl")]
    sys.modules["rl"] = rl
    xenv = types.ModuleType("xenv")
    xenv.__path__ = [str(source / "xenv")]
    sys.modules["xenv"] = xenv
    # Avoid the obsolete yacs config entrypoint; the actual sensor/model files stay unchanged.
    sensors = importlib.import_module("xenv.sensors")
    xenv.ImageGoalSensorV2 = sensors.ImageGoalSensorV2
    xenv.PoseSensor = sensors.PoseSensor
    importlib.import_module("xenv.measures")
    return importlib.import_module("rl.policy.navnet_policy").NavNetPolicy


def load_legacy_checkpoint(path):
    import habitat.config.default
    import torch

    # Only obsolete serialized configuration metadata needs this class alias.
    class LegacyConfig(dict):
        pass

    habitat.config.default.Config = LegacyConfig
    return torch.load(path, map_location="cpu")
