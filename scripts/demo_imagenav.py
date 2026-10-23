"""The original ZER ImageNav policy on public scenes, with fixed image-goal input."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time

from legacy_policy_loader import load_zer_policy_class


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
    from habitat.config.default_structured_configs import LabSensorConfig, MeasurementConfig
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
    Policy = load_zer_policy_class(root)
    from xenv.utils import ResizeCenterCropper

    @dataclass
    class GoalConfig(LabSensorConfig):
        type: str = "ImageGoalSensorV2"
        SAMPLING_TYPE: str = "uniform"
        CHANNELS: str = "rgb"

    @dataclass
    class ViewMatchConfig(MeasurementConfig):
        type: str = "ViewMatch"
        ANGLE_THRESHOLD: float = 25.0
        VIEW_WEIGHT: float = 0.5

    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    with read_write(config):
        cfg = config.habitat
        cfg.seed = args.seed
        cfg.dataset.split = "val"
        cfg.environment.max_episode_steps = 1000
        cfg.environment.iterator_options.shuffle = False
        cfg.simulator.habitat_sim_v0.gpu_device_id = args.gpu
        cfg.simulator.habitat_sim_v0.allow_sliding = False
        cfg.simulator.turn_angle = 30
        rgb = cfg.simulator.agents.main_agent.sim_sensors.rgb_sensor
        rgb.height = rgb.width = 128
        rgb.hfov = 90
        cfg.simulator.agents.main_agent.sim_sensors = {"rgb_sensor": rgb}
        cfg.task.lab_sensors = {"imagegoalv2": GoalConfig()}
        cfg.task.goal_sensor_uuid = "imagegoalv2"
        cfg.task.measurements.success.success_distance = 1.0
        cfg.task.measurements.view_angle = MeasurementConfig(type="ViewAngle")
        cfg.task.measurements.view_match = ViewMatchConfig()
        # Config-key aliases used by the unchanged original evaluator, never policy features.
        OmegaConf.set_struct(cfg.task, False)
        cfg.task.GOAL_SENSOR_UUID = "imagegoalv2"
        cfg.task.SUCCESS_DISTANCE = 1.0
    (args.output / "task.yaml").write_text(OmegaConf.to_yaml(config))
    dataset = habitat.make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    dataset.episodes = dataset.episodes[:args.episodes]
    assert len(dataset.episodes) == args.episodes
    rows = []
    start = time.monotonic()
    with habitat.Env(config=config.habitat, dataset=dataset) as env:
        keys = ("rgb", "imagegoalv2")
        space = gym.spaces.Dict({k: env.observation_space.spaces[k] for k in keys})
        policy = Policy(space, env.action_space, hidden_size=128, num_recurrent_layers=2,
                        rnn_type="GRU", backbone="fast_resnet9", normalize_visual_inputs=True,
                        obs_transform=ResizeCenterCropper(size=(128, 128)),
                        visual_encoder_embedding_size=512, visual_obs_inputs=["rgb"],
                        visual_encoder_init="", goal_embedding_size=128).to(device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = {k.removeprefix("actor_critic."): v for k, v in checkpoint["state_dict"].items()
                 if k.startswith("actor_critic.")}
        incompatible = policy.load_state_dict(state, strict=True)
        policy.eval()
        (args.output / "checkpoint_load.json").write_text(json.dumps({
            "missing_keys": incompatible.missing_keys, "unexpected_keys": incompatible.unexpected_keys,
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "source_commit": "dfe43e8ff9da74f27fe4db9ef311d9d40aeed67a",
            "dependency_commit": "ac937fde9e14a47968eaf221857eb4e65f48383e",
            "parameters": sum(p.numel() for p in policy.parameters()),
            "policy_observations": keys, "online_pose_or_gps_input": False,
            "dataset_domain": "official-public-test-scenes-not-Gibson",
            "goal_source": "original ImageGoalSensorV2 renders one fixed goal image as task instruction",
        }, indent=2))
        for index in range(args.episodes):
            obs = env.reset()
            goal = obs["imagegoalv2"].copy()
            goal_bytes = goal.astype(np.uint8)
            Image.fromarray(goal_bytes).save(args.output / f"episode_{index:02d}.goal.png")
            hidden = torch.zeros(policy.net.num_recurrent_layers, 1, 128, device=device)
            previous = torch.zeros(1, 1, dtype=torch.long, device=device)
            mask = torch.zeros(1, 1, device=device)
            actions = []
            writer = imageio.get_writer(str(args.output / f"episode_{index:02d}.mp4"), fps=10,
                                       codec="libx264", macro_block_size=1)
            try:
                while not env.episode_over:
                    assert set(obs) == set(keys)
                    assert np.array_equal(obs["imagegoalv2"], goal), "Goal image changed within episode"
                    batch = {k: torch.as_tensor(obs[k], device=device).unsqueeze(0) for k in keys}
                    with torch.no_grad():
                        value, action, _, hidden = policy.act(batch, hidden, previous, mask, deterministic=False)
                        assert torch.isfinite(value).all()
                    action = int(action.item())
                    frame = Image.new("RGB", (512, 304), "#17212b")
                    frame.paste(Image.fromarray(obs["rgb"]).resize((256, 256)), (0, 48))
                    frame.paste(Image.fromarray(goal_bytes).resize((256, 256)), (256, 48))
                    draw = ImageDraw.Draw(frame)
                    draw.text((8, 6), "ZER ImageNav | current RGB + FIXED goal image | NO GPS", fill="white")
                    draw.text((8, 25), f"CURRENT VIEW   step={len(actions)} action={action}", fill="white")
                    draw.text((270, 25), "GOAL PHOTO (constant)", fill="white")
                    writer.append_data(np.asarray(frame))
                    obs = env.step(action)
                    actions.append(action)
                    previous.fill_(action)
                    mask.fill_(1.0)
            finally:
                writer.close()
            row = {"episode_id": env.current_episode.episode_id, "scene_id": env.current_episode.scene_id,
                   **env.get_metrics(), "num_steps": len(actions), "actions": actions,
                   "stopped": bool(actions and actions[-1] == 0), "goal_constant_all_steps": True,
                   "goal_sha256": hashlib.sha256(goal.tobytes()).hexdigest()}
            rows.append(row)
            with (args.output / "episodes.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "actions"}), flush=True)
    summary = {"status": "complete", "completed_episodes": len(rows), "expected_episodes": args.episodes,
               "dataset_domain": "official-public-test-scenes-not-Gibson",
               "success": float(np.mean([r["success"] for r in rows])),
               "spl": float(np.mean([r["spl"] for r in rows])),
               "view_match": float(np.mean([r["view_match"] for r in rows])),
               "elapsed_seconds": time.monotonic() - start}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
