"""Two-node machine allocation for the unchanged 200M EmbCLIP comparisons."""

from allenact.base_abstractions.experiment_config import MachineParams
from projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig,
)


class Formal8MachineMixin:
    def machine_params(self, mode="train", **kwargs):
        if mode == "valid":
            return MachineParams(nprocesses=0, devices=[-1])
        params = ObjectNavRoboThorClipRGBPPOExperimentConfig.machine_params(
            self, mode=mode, **kwargs
        )
        if mode != "train":
            return params
        assert self.num_train_processes == 60
        assert len(params.devices) == len(params.sampler_devices) == 4
        machine_id = kwargs.get("machine_id")
        if machine_id is not None:
            assert machine_id in (0, 1)
            local_ids = list(range(4 * machine_id, 4 * (machine_id + 1)))
        else:
            local_ids = None
        return MachineParams(
            nprocesses=(8, 8, 8, 8, 7, 7, 7, 7),
            devices=params.devices * 2,
            sampler_devices=params.sampler_devices * 2,
            sensor_preprocessor_graph=params.sensor_preprocessor_graph,
            local_worker_ids=local_ids,
        )
