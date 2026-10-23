"""Original AllenAct RGB-D ResNetGRU policy on public RoboTHOR val episodes."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--display", default="1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "vendor/allenact"))

    import gym
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from ai2thor.controller import Controller
    from allenact.base_abstractions.misc import Memory
    from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
    from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
    from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
    from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
    from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
    from projects.objectnav_baselines.models.object_nav_models import ResnetTensorObjectNavActorCritic

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.hub.set_dir(str(root / "assets/torch-hub"))
    cache = root / "assets/ai2thor"
    cache.mkdir(exist_ok=True)
    # Redirect the simulator's download cache, without changing its rendering or actions.
    Controller.base_dir = property(lambda self: str(cache))
    categories = tuple(sorted(["AlarmClock", "Apple", "BaseballBat", "BasketBall", "Bowl", "GarbageCan",
                               "HousePlant", "Laptop", "Mug", "SprayBottle", "Television", "Vase"]))
    sensors = [RGBSensorThor(height=224, width=224, use_resnet_normalization=True, uuid="rgb_lowres"),
               DepthSensorThor(height=224, width=224, use_normalization=True, uuid="depth_lowres"),
               GoalObjectTypeThorSensor(object_types=categories)]
    sensor_keys = {s.uuid for s in sensors}
    assert sensor_keys == {"rgb_lowres", "depth_lowres", "goal_object_type_ind"}
    spaces = gym.spaces.Dict({"rgb_resnet": gym.spaces.Box(-np.inf, np.inf, (512, 7, 7), dtype=np.float32),
                              "depth_resnet": gym.spaces.Box(-np.inf, np.inf, (512, 7, 7), dtype=np.float32),
                              "goal_object_type_ind": gym.spaces.Discrete(len(categories))})
    policy = ResnetTensorObjectNavActorCritic(
        gym.spaces.Discrete(len(ObjectNavTask.class_action_names())), spaces,
        goal_sensor_uuid="goal_object_type_ind", rgb_resnet_preprocessor_uuid="rgb_resnet",
        depth_resnet_preprocessor_uuid="depth_resnet", hidden_size=512, goal_dims=32).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # The original AllenAct loader handles its own documented state_encoder key rename.
    incompatible = policy.load_state_dict(checkpoint["model_state_dict"])
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    policy.eval()
    preprocessors = [ResNetPreprocessor([name + "_lowres"], name + "_resnet", 224, 224, 7, 7, 512,
                                       pool=False, device=device).to(device) for name in ("rgb", "depth")]
    files = sorted(args.dataset.glob("*.json.gz"))
    assert files, args.dataset
    all_episodes = json.loads(gzip.decompress(files[0].read_bytes()))
    if isinstance(all_episodes, dict):
        all_episodes = all_episodes["episodes"]
    episodes, seen = [], set()
    for episode in all_episodes:
        if episode["object_type"] not in seen:
            episodes.append(episode)
            seen.add(episode["object_type"])
        if len(episodes) == args.episodes:
            break
    assert len(episodes) == args.episodes
    (args.output / "selected_episodes.json").write_text(json.dumps(episodes, indent=2))
    receipt = {"missing_keys": incompatible.missing_keys, "unexpected_keys": incompatible.unexpected_keys,
               "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               "policy_observations": sorted(sensor_keys), "online_pose_or_gps_input": False,
               "source_commit": "f00445e4ae8724ccc001b3300af5c56a7f882614",
               "simulator_commit": "bad5bc2b250615cb766ffb45d455c211329af17e",
               "dataset_file": str(files[0]), "dataset_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
               "selection": "first validation scene; first episode of first three distinct categories; no outcome filtering",
               "parameters": sum(p.numel() for p in policy.parameters()),
               "action_names": ObjectNavTask.class_action_names(), "checkpoint_training_steps": checkpoint["total_steps"]}
    (args.output / "checkpoint_load.json").write_text(json.dumps(receipt, indent=2))
    vertical_fov = math.degrees(2 * math.atan(math.tan(math.radians(79) / 2) * 300 / 400))
    start = time.monotonic()
    env = RoboThorEnvironment(commit_id=receipt["simulator_commit"], scene=episodes[0]["scene"],
                              x_display=args.display, width=400, height=300, fieldOfView=vertical_fov,
                              applyActionNoise=True, renderDepthImage=True, include_private_scenes=False)
    rows = []
    try:
        for index, episode in enumerate(episodes):
            env.reset(episode["scene"])
            assert env.teleport(episode["initial_position"], {"x": 0, "y": episode["initial_orientation"], "z": 0},
                                horizon=episode["initial_horizon"])
            info = dict(episode, mirrored=False)
            task = ObjectNavTask(env, sensors, info, max_steps=500,
                                 reward_configs={"step_penalty": -0.01, "goal_success_reward": 10.0,
                                                 "failed_stop_reward": 0.0, "shaping_weight": 0.0})
            memory = Memory({"single_belief": (torch.zeros(1, 1, 512, device=device), 1)})
            previous = torch.zeros(1, 1, dtype=torch.long, device=device)
            mask = torch.zeros(1, 1, 1, device=device)
            actions = []
            writer = imageio.get_writer(str(args.output / f"episode_{index:02d}.mp4"), fps=10,
                                       codec="libx264", macro_block_size=1)
            try:
                while not task.is_done():
                    obs = task.get_observations()
                    assert set(obs) == sensor_keys
                    tensors = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
                    with torch.no_grad():
                        features = {p.uuid: p.process(tensors).unsqueeze(0) for p in preprocessors}
                        features["goal_object_type_ind"] = tensors["goal_object_type_ind"].reshape(1, 1)
                        output, memory = policy(features, memory, previous, mask)
                        assert torch.isfinite(output.values).all()
                        action = int(output.distributions.sample().item())
                    frame = Image.new("RGB", (400, 348), "#17212b")
                    frame.paste(Image.fromarray(env.current_frame), (0, 48))
                    draw = ImageDraw.Draw(frame)
                    draw.text((8, 6), "RoboTHOR | RGB-D + target category | NO GPS", fill="white")
                    draw.text((8, 25), f"Find {episode['object_type']} | step={len(actions)} | {task.class_action_names()[action]}", fill="white")
                    writer.append_data(np.asarray(frame))
                    task.step(action)
                    actions.append(action)
                    previous.fill_(action)
                    mask.fill_(1.0)
            finally:
                writer.close()
            metrics = task.metrics()
            row = {"episode_id": episode["id"], "scene_id": episode["scene"], "target": episode["object_type"],
                   **metrics, "num_steps": len(actions), "actions": actions,
                   "stopped": bool(actions and task.class_action_names()[actions[-1]] == "End")}
            assert all(np.isfinite(row[k]) for k in ("success", "spl"))
            rows.append(row)
            with (args.output / "episodes.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
    finally:
        env.stop()
    summary = {"status": "complete", "completed_episodes": len(rows), "expected_episodes": args.episodes,
               "dataset_domain": "RoboTHOR-public-validation-first-scene-three-categories",
               "success": float(np.mean([r["success"] for r in rows])),
               "spl": float(np.mean([r["spl"] for r in rows])), "elapsed_seconds": time.monotonic() - start}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
