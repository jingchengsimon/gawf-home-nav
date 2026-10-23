"""Evaluate the complete official PointNav policy on disjoint per-GPU episode shards.

Inputs: pinned Habitat sources, original checkpoint and an official task config.
Outputs: exact inputs, strict-load receipt, per-episode JSONL, summary and RGB videos.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import time

from pointnav_common import mean_metrics, policy_state_dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="val")
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument("--task-config", default="benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    parser.add_argument("--dataset-domain", default="official-public-test-scenes")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size or args.episodes <= 0:
        parser.error("Invalid shard or episode budget")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)

    import gym
    import habitat
    from habitat.config import read_write
    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
    import imageio.v2 as imageio
    import numpy as np
    from omegaconf import OmegaConf
    import torch

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU evaluation requested but CUDA unavailable")
    device = torch.device(f"cuda:{args.rank}")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(args.seed)
    config = habitat.get_config(args.task_config)
    with read_write(config):
        config.habitat.seed = args.seed
        config.habitat.dataset.split = args.split
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.rank
        config.habitat.environment.iterator_options.shuffle = False
    (args.output / "task.yaml").write_text(OmegaConf.to_yaml(config))
    dataset = habitat.make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    selected = dataset.episodes[args.rank::args.world_size][:args.episodes]
    if len(selected) != args.episodes:
        raise RuntimeError(f"Need {args.episodes} unique episodes; shard has {len(selected)}")
    dataset.episodes = selected
    source_root = Path(habitat.__file__).resolve().parents[2]
    command = {
        "args": vars(args), "checkpoint_sha256": sha256(args.checkpoint),
        "dataset_domain": args.dataset_domain, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "habitat": habitat.__version__,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "policy_observations": ["depth", "pointgoal_with_gps_compass"],
        "episode_file_sha256": sha256(Path(config.habitat.dataset.data_path.format(split=args.split))),
        "scene_sha256": {scene: sha256(Path(scene)) for scene in {ep.scene_id for ep in selected}},
        "sampled_actions": True,
        "episodes": [{"scene_id": ep.scene_id, "episode_id": ep.episode_id} for ep in selected],
    }
    (args.output / "command.json").write_text(json.dumps(command, indent=2, default=str))
    results = []
    total_steps = 0
    policy_seconds = 0.0
    env_seconds = 0.0
    start_time = time.perf_counter()
    with habitat.Env(config=config.habitat, dataset=dataset) as env:
        policy_space = gym.spaces.Dict({key: env.observation_space.spaces[key] for key in
                                       ("depth", "pointgoal_with_gps_compass")})
        policy = PointNavResNetPolicy(
            observation_space=policy_space, action_space=env.action_space,
            hidden_size=512, num_recurrent_layers=2, rnn_type="LSTM", backbone="resnet50",
            normalize_visual_inputs=False,
        ).to(device)
        original = torch.load(args.checkpoint, map_location="cpu")
        incompatible = policy.load_state_dict(policy_state_dict(original["state_dict"]), strict=True)
        receipt = {"missing_keys": incompatible.missing_keys,
                   "unexpected_keys": incompatible.unexpected_keys,
                   "parameter_count": sum(p.numel() for p in policy.parameters()),
                   "checkpoint_sha256": command["checkpoint_sha256"]}
        (args.output / "checkpoint_load.json").write_text(json.dumps(receipt, indent=2))
        policy.eval()
        with (args.output / "episodes.jsonl").open("x") as episode_log:
            for episode_index in range(args.episodes):
                observation = env.reset()
                episode = env.current_episode
                hidden = torch.zeros(1, policy.net.num_recurrent_layers, 512, device=device)
                prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
                mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
                frames = []
                actions = []
                while not env.episode_over:
                    if episode_index < args.videos:
                        frames.append(np.asarray(observation["rgb"]).copy())
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    batch = {key: torch.as_tensor(observation[key], device=device).unsqueeze(0)
                             for key in ("depth", "pointgoal_with_gps_compass")}
                    with torch.no_grad():
                        data = policy.act(batch, hidden, prev_action, mask, deterministic=False)
                    torch.cuda.synchronize()
                    policy_seconds += time.perf_counter() - tick
                    hidden = data.rnn_hidden_states
                    prev_action.copy_(data.actions)
                    mask.fill_(True)
                    action = int(data.env_actions.item())
                    tick = time.perf_counter()
                    observation = env.step(action)
                    env_seconds += time.perf_counter() - tick
                    actions.append(action)
                    total_steps += 1
                row = {"episode_id": episode.episode_id, "scene_id": episode.scene_id,
                       **env.get_metrics(), "num_steps": len(actions),
                       "stopped": bool(actions and actions[-1] == 0), "actions": actions}
                mean_metrics([row])
                episode_log.write(json.dumps(row, allow_nan=False) + "\n")
                episode_log.flush()
                if frames:
                    imageio.mimsave(args.output / f"episode_{episode_index:02d}.mp4", frames,
                                    fps=10, codec="libx264", macro_block_size=1)
                results.append(row)
                logging.info("Completed rank=%s episode=%s/%s steps=%s success=%s SPL=%s",
                             args.rank, episode_index + 1, args.episodes, len(actions),
                             row["success"], row["spl"])
    elapsed = time.perf_counter() - start_time
    summary = {"status": "complete", "completed_episodes": len(results),
               "expected_episodes": args.episodes, "dataset_domain": args.dataset_domain,
               "metrics": mean_metrics(results), "total_steps": total_steps,
               "wall_seconds": elapsed, "policy_seconds": policy_seconds,
               "env_seconds": env_seconds, "end_to_end_steps_per_second": total_steps / elapsed,
               "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    logging.info("Evaluation complete: %s", summary)


if __name__ == "__main__":
    main()
