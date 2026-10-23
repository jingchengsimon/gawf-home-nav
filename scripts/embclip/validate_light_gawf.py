"""Check parameter matching, reset, and gradients for light GaWF models."""

import json
import types

import torch

from allenact.base_abstractions.misc import Memory
from projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig as OfficialConfig,
)

from scripts.embclip.light_gru1_config import LightGRU1Config
from scripts.embclip.light_gawf1_config import LightGaWF1Config
from scripts.embclip.light_gawf3_config import LightGaWF3Config


def memory_for(model):
    memory = Memory()
    for key, (dims, dtype) in model.recurrent_memory_specification.items():
        shape = tuple(1 if name == "sampler" else size for name, size in dims)
        sampler_dim = [name for name, _ in dims].index("sampler")
        memory.check_append(key, torch.zeros(*shape, dtype=dtype), sampler_dim)
    return memory


def main():
    torch.manual_seed(7)
    counts = {}
    for config_class in (LightGRU1Config, LightGaWF1Config, LightGaWF3Config):
        config = config_class(headless=True, train_gpu_ids=[0], num_train_processes=16)
        graph = OfficialConfig.machine_params(config, mode="train").sensor_preprocessor_graph
        model = config.create_model(sensor_preprocessor_graph=graph)
        counts[config.MODEL] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pipeline = config.training_pipeline()
        stage = pipeline.pipeline_stages[0]
        assert stage.max_stage_steps == 1_000_000
        assert stage.training_settings.save_interval == 100_000
        if config.MODEL == "gru1":
            continue
        assert model.num_recurrent_layers == (1 if config.MODEL == "gawf1" else 3)
        width = model.goal_visual_encoder.output_dims
        model.forward_encoder = types.MethodType(lambda self, obs: obs["fused"], model)
        features = torch.randn(2, 1, width)
        actions = torch.zeros(2, 1, dtype=torch.long)
        masks = torch.tensor([[[1.0]], [[0.0]]])
        model.eval()
        with torch.no_grad():
            joined, _ = model({"fused": features}, memory_for(model), actions, masks)
            fresh, _ = model({"fused": features[1:]}, memory_for(model), actions[1:], masks[1:])
        assert torch.allclose(joined.distributions.logits[1:], fresh.distributions.logits, atol=1e-6)
        assert torch.allclose(joined.values[1:], fresh.values, atol=1e-6)
        model.train()
        train_features = features.detach().clone().requires_grad_(True)
        output, final_memory = model(
            {"fused": train_features}, memory_for(model), actions, torch.ones_like(masks)
        )
        (output.values.sum() + output.distributions.logits[..., 0].sum()).backward()
        assert torch.isfinite(train_features.grad).all()
        core = model.state_encoders["single_belief"]
        gate = core.U if config.MODEL == "gawf1" else core.U_layers[0]
        assert gate.grad is not None and gate.grad.abs().sum() > 0
        assert final_memory.tensor("policy_feedback").shape == (1, model.action_space.n)
    assert counts["gru1"] == 3_480_775, counts
    assert counts["gawf1"] == 3_479_807, counts
    assert counts["gawf3"] == 3_488_347, counts
    print(json.dumps({"status": "pass", "trainable_parameters": counts,
                      "max_parameter_difference_fraction": max(
                          abs(counts[model] - counts["gru1"]) / counts["gru1"]
                          for model in ("gawf1", "gawf3")),
                      "reset_and_gradient": "pass", "save_interval_steps": 100_000}))


if __name__ == "__main__":
    main()
