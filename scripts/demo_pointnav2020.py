"""Official 2020 recurrent policy and noise settings on public test scenes.

The fixed initial target is a task instruction. Only RGB-D, that instruction,
previous actions and recurrent state enter the original published policy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from legacy_policy_loader import load_2020_policy_class, load_legacy_checkpoint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    assert 1 <= args.episodes <= 10
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]

    import gym
    import habitat
    from habitat.config import read_write
    from habitat.config.default_structured_configs import PointGoalSensorConfig
    import habitat_sim
    import imageio.v2 as imageio
    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image, ImageDraw
    import torch

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    with read_write(config):
        cfg = config.habitat
        cfg.seed = args.seed
        cfg.dataset.split = "val"
        cfg.environment.max_episode_steps = 500
        cfg.environment.iterator_options.shuffle = False
        sim = cfg.simulator
        sim.habitat_sim_v0.gpu_device_id = args.gpu
        sim.habitat_sim_v0.allow_sliding = False
        sim.turn_angle = 30
        agent = sim.agents.main_agent
        agent.height = 0.88
        agent.radius = 0.18
        for key in ("rgb_sensor", "depth_sensor"):
            sensor = agent.sim_sensors[key]
            sensor.height = 360
            sensor.width = 640
            sensor.hfov = 70
            sensor.position = [0, 0.88, 0]
        rgb = agent.sim_sensors.rgb_sensor
        rgb.noise_model = "GaussianNoiseModel"
        rgb.noise_model_kwargs = {"intensity_constant": 0.1}
        depth = agent.sim_sensors.depth_sensor
        depth.min_depth = 0.1
        depth.max_depth = 10.0
        depth.normalize_depth = True
        depth.noise_model = "RedwoodDepthNoiseModel"
        cfg.task.lab_sensors = {"pointgoal_sensor": PointGoalSensorConfig()}
        cfg.task.goal_sensor_uuid = "pointgoal"
        cfg.task.measurements.success.success_distance = 0.36
    (args.output / "task.yaml").write_text(OmegaConf.to_yaml(config))
    dataset = habitat.make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    dataset.episodes = dataset.episodes[:args.episodes]
    assert len(dataset.episodes) == args.episodes

    def noisy_actions(env):
        spec = habitat_sim.PyRobotNoisyActuationSpec
        controls = {1: ("pyrobot_noisy_move_forward", 0.25),
                    2: ("pyrobot_noisy_turn_left", 30), 3: ("pyrobot_noisy_turn_right", 30)}
        action_space = env.sim.get_agent(0).agent_config.action_space
        for key, (name, amount) in controls.items():
            action_space[key] = habitat_sim.ActionSpec(name, spec(
                amount=amount, robot="LoCoBot", controller="Proportional", noise_multiplier=0.5))

    policy_keys = ("rgb", "depth", "pointgoal")
    start = time.monotonic()
    rows = []
    with habitat.Env(config=config.habitat, dataset=dataset) as env:
        space = gym.spaces.Dict({k: env.observation_space.spaces[k] for k in policy_keys})
        Policy = load_2020_policy_class(root)
        policy = Policy(space, env.action_space, goal_sensor_uuid="pointgoal", hidden_size=512,
                        num_recurrent_layers=2, rnn_type="LSTM", backbone="resnet50",
                        normalize_visual_inputs=True).to(device).eval()
        checkpoint = load_legacy_checkpoint(args.checkpoint)
        state = {k.removeprefix("actor_critic."): v for k, v in checkpoint["state_dict"].items()
                 if k.startswith("actor_critic.")}
        incompatible = policy.load_state_dict(state, strict=True)
        (args.output / "checkpoint_load.json").write_text(json.dumps({
            "missing_keys": incompatible.missing_keys, "unexpected_keys": incompatible.unexpected_keys,
            "checkpoint_sha256": digest(args.checkpoint), "policy_observations": policy_keys,
            "parameters": sum(p.numel() for p in policy.parameters()),
            "source_commit": "959bd45431edd8024832a877bdc8218015d97a7e",
            "dataset_domain": "official-public-test-scenes-not-Gibson",
            "actuation": "official-pyrobot-noisy-controls-LoCoBot-Proportional-0.5",
            "online_pose_or_gps_input": False,
        }, indent=2))
        for index in range(args.episodes):
            obs = env.reset()
            noisy_actions(env)
            initial_goal = obs["pointgoal"].copy()
            hidden = torch.zeros(policy.net.num_recurrent_layers, 1, 512, device=device)
            previous = torch.zeros(1, 1, dtype=torch.long, device=device)
            mask = torch.zeros(1, 1, device=device)
            actions, trace = [], []
            writer = imageio.get_writer(str(args.output / f"episode_{index:02d}.mp4"), fps=10,
                                       codec="libx264", macro_block_size=1)
            try:
                while not env.episode_over:
                    assert set(obs) == set(policy_keys), set(obs)
                    assert np.array_equal(obs["pointgoal"], initial_goal), "Target instruction changed"
                    batch = {k: torch.as_tensor(obs[k], device=device).unsqueeze(0) for k in policy_keys}
                    with torch.no_grad():
                        _, action, _, hidden = policy.act(batch, hidden, previous, mask, deterministic=False)
                    action = int(action.item())
                    image = Image.fromarray(obs["rgb"])
                    frame = Image.new("RGB", (640, 408), "#17212b")
                    frame.paste(image, (0, 48))
                    draw = ImageDraw.Draw(frame)
                    draw.text((8, 6), "PointNav 2020 | RGB-D + FIXED initial goal | NO GPS/Compass", fill="white")
                    draw.text((8, 25), f"initial goal={initial_goal.round(3).tolist()}  step={len(actions)}  action={action}", fill="white")
                    writer.append_data(np.asarray(frame))
                    trace.append({"step": len(actions), "pointgoal": obs["pointgoal"].tolist(), "action": action})
                    obs = env.step(action)
                    actions.append(action)
                    previous.fill_(action)
                    mask.fill_(1.0)
            finally:
                writer.close()
            row = {"episode_id": env.current_episode.episode_id, "scene_id": env.current_episode.scene_id,
                   **env.get_metrics(), "num_steps": len(actions), "stopped": bool(actions and actions[-1] == 0),
                   "initial_goal": initial_goal.tolist(), "goal_constant_all_steps": True, "actions": actions}
            rows.append(row)
            (args.output / f"episode_{index:02d}.trace.json").write_text(json.dumps(trace))
            with (args.output / "episodes.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
    summary = {"status": "complete", "completed_episodes": len(rows), "expected_episodes": args.episodes,
               "dataset_domain": "official-public-test-scenes-not-Gibson",
               "success": float(np.mean([r["success"] for r in rows])),
               "spl": float(np.mean([r["spl"] for r in rows])), "elapsed_seconds": time.monotonic() - start}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
