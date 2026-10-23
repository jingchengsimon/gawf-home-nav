"""CPU-only structural and recurrence checks before any formal GPU submission."""

import json
import types
from unittest.mock import patch

import torch

from allenact.base_abstractions.misc import Memory
from projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig as OfficialConfig,
)

from scripts.embclip.formal_gru1_config import FormalGRU1Config
from scripts.embclip.formal_gru3_config import FormalGRU3Config
from scripts.embclip.formal_gawf3_config import FormalGaWF3Config


def new_memory(model, samplers):
    memory = Memory()
    for key, (dims, dtype) in model.recurrent_memory_specification.items():
        shape = tuple(samplers if name == "sampler" else size for name, size in dims)
        sampler_dim = [name for name, _ in dims].index("sampler")
        memory.check_append(key, torch.zeros(*shape, dtype=dtype), sampler_dim)
    return memory


def main():
    torch.manual_seed(7)
    counts = {}
    models = {}
    for config_class in (FormalGRU1Config, FormalGRU3Config, FormalGaWF3Config):
        config = config_class(train_gpu_ids=[0, 1], num_train_processes=60)
        graph = OfficialConfig.machine_params(config, mode="train").sensor_preprocessor_graph
        model = config.create_model(sensor_preprocessor_graph=graph)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        counts[config.MODEL] = count
        models[config.MODEL] = model
        assert config.training_pipeline().pipeline_stages[0].max_stage_steps == 200_000_000
    assert counts["gru1"] == 3_480_775, counts
    for variant in ("gru3", "gawf3"):
        assert abs(counts[variant] - counts["gru1"]) / counts["gru1"] < 0.005, counts

    model = models["gawf3"].eval()
    features = torch.randn(2, 1, model.goal_visual_encoder.output_dims)
    model.forward_encoder = types.MethodType(lambda self, obs: obs["fused"], model)
    initial = new_memory(model, 1)
    masks = torch.tensor([[[1.0]], [[0.0]]])
    actions = torch.zeros(2, 1, dtype=torch.long)
    with torch.no_grad():
        whole, end_whole = model({"fused": features}, initial, actions, masks)
        one, _ = model(
            {"fused": features[1:]}, new_memory(model, 1), actions[1:], masks[1:]
        )
    assert torch.allclose(whole.distributions.logits[1:], one.distributions.logits, atol=1e-6)
    assert torch.allclose(whole.values[1:], one.values, atol=1e-6)
    assert end_whole.tensor("single_belief").shape == (3, 1, 653)
    assert end_whole.tensor("policy_feedback").shape == (1, model.action_space.n)

    model.train()
    train_features = features.detach().clone().requires_grad_(True)
    continuous_masks = torch.ones_like(masks)
    trained, _ = model(
        {"fused": train_features}, new_memory(model, 1), actions, continuous_masks
    )
    loss = trained.values.sum() + trained.distributions.logits[..., 0].sum()
    loss.backward()
    assert torch.isfinite(train_features.grad).all()
    assert model.state_encoders["single_belief"].U_layers[0].grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        direct, _ = model(
            {"fused": features}, new_memory(model, 1), actions, continuous_masks
        )
    assert torch.allclose(trained.values.detach(), direct.values, atol=1e-6)

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.get_device_capability", return_value=(8, 0)
    ):
        distributed = FormalGRU1Config(
            train_gpu_ids=[0, 1, 2, 3], num_train_processes=60
        )
        global_params = distributed.machine_params("train")
        assert len(global_params.devices) == 16
        assert sum(global_params.nprocesses) == 60
        local_ids = [distributed.machine_params("train", machine_id=i).local_worker_ids for i in range(4)]
        assert local_ids == [list(range(i * 4, (i + 1) * 4)) for i in range(4)]

    print(json.dumps({"trainable_parameters": counts, "reset_check": "pass", "checkpoint_gradient_check": "pass", "global_environments": 60, "distributed_worker_ids": local_ids}, sort_keys=True))


if __name__ == "__main__":
    main()
