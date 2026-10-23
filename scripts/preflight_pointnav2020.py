"""GPU validation of the actual legacy policy, sensors, noise and simulator API."""

import argparse
import hashlib
import json
import os
import random
import time

from pointnav2020_runtime import ROOT, OBSERVATIONS, load_config, run_paths
from pointnav2020_compat import patch_records

import habitat
import numpy as np
import torch
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    args = parser.parse_args()
    results, logs = run_paths(args.experiment_id)
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    config = load_config(args.experiment_id, split="val")
    checkpoint_path = ROOT / "assets/checkpoints/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth"
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert digest == "9f8fa1a595d6b2bcd5b85c934e3638b10a8aad6a465c2396b168cb62b5892301"
    dataset = habitat.make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes = dataset.episodes[:1]
    started = time.monotonic()
    env = habitat.Env(config=config.TASK_CONFIG, dataset=dataset)
    try:
        observation = env.reset()
        assert set(observation) == OBSERVATIONS
        goal = observation["pointgoal"].copy()
        native = env.sim._sim
        assert native.pathfinder.is_loaded
        ns = native.pathfinder.nav_mesh_settings
        assert np.isclose(ns.agent_radius, 0.18) and np.isclose(ns.agent_height, 0.88)
        specs = env.sim.sim_config.agents[0].sensor_specifications
        assert {s.noise_model for s in specs} == {"GaussianNoiseModel", "RedwoodDepthNoiseModel"}
        assert all(np.isclose(s.orientation[0], -0.3490659) for s in specs)
        assert not env.sim.sim_config.sim_cfg.allow_sliding
        controls = env.sim.sim_config.agents[0].action_space
        assert controls[1].name == "pyrobot_noisy_move_forward"
        assert controls[2].name == "pyrobot_noisy_turn_left"
        assert controls[3].name == "pyrobot_noisy_turn_right"
        sample = env.sim.sensor_suite.get_observations(native.get_sensor_observations())
        assert not np.array_equal(observation["rgb"], sample["rgb"])
        assert not np.array_equal(observation["depth"], sample["depth"])
        assert np.isfinite(sample["depth"]).all()
        assert sample["depth"].min() >= 0 and sample["depth"].max() <= 1
        policy = PointNavResNetPolicy(env.observation_space, env.action_space, goal_sensor_uuid="pointgoal",
                                     hidden_size=512, num_recurrent_layers=2, rnn_type="LSTM",
                                     backbone="resnet50", normalize_visual_inputs=True).cuda().eval()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = {k[len("actor_critic."):]: v for k, v in checkpoint["state_dict"].items()}
        incompatible = policy.load_state_dict(state, strict=True)
        hidden = torch.zeros(policy.net.num_recurrent_layers, 1, 512, device="cuda")
        previous = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        masks = torch.zeros(1, 1, device="cuda")
        steps = 0
        for _ in range(16):
            assert set(observation) == OBSERVATIONS
            assert np.array_equal(observation["pointgoal"], goal)
            batch = {k: torch.as_tensor(v, device="cuda").unsqueeze(0) for k, v in observation.items()}
            with torch.no_grad():
                value, action, logp, hidden = policy.act(batch, hidden, previous, masks, deterministic=False)
            assert torch.isfinite(value).all() and torch.isfinite(logp).all() and torch.isfinite(hidden).all()
            observation = env.step(int(action.item()))
            previous.copy_(action)
            masks.fill_(1)
            steps += 1
            if env.episode_over:
                break
        row = {"status": "pass", "experiment_id": args.experiment_id,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "scene_id": env.current_episode.scene_id.split("/gibson/")[-1],
               "policy_observations": sorted(OBSERVATIONS), "goal_constant": True,
               "strict_checkpoint_load": {"missing": incompatible.missing_keys, "unexpected": incompatible.unexpected_keys},
               "checkpoint_sha256": digest, "steps": steps,
               "noisy_rgb_depth_verified": True, "allow_sliding": False,
               "navmesh_agent_radius": ns.agent_radius, "navmesh_agent_height": ns.agent_height,
               "patches": patch_records(), "elapsed_s": time.monotonic() - started}
    finally:
        env.close()
    (results / "summary.json").write_text(json.dumps(row, indent=2, allow_nan=False))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
