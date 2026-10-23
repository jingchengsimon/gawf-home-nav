"""Record one paired RoboTHOR RGB episode from two frozen EmbCLIP policies.

Input: a registered experiment manifest with two checkpoint identities and one val episode ID.
Outputs: two policy rollout videos, their action/metric receipts, and one synchronized MP4.
The script runs only on SJC; checkpoint actions use RGB and target category observations.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor/embclip-allenact"))


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def find_episode(dataset_path, episode_id):
    with gzip.open(dataset_path, "rt") as stream:
        episodes = json.load(stream)
    matches = [episode for episode in episodes if episode["id"] == episode_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one episode {episode_id}, found {len(matches)}")
    return matches[0]


def initial_memory(model, device, torch):
    from allenact.base_abstractions.misc import Memory

    entries = {}
    for name, (dims, dtype) in model.recurrent_memory_specification.items():
        shape = tuple(1 if size is None else size for _, size in dims)
        sampler_dim = [label for label, _ in dims].index("sampler")
        entries[name] = (torch.zeros(shape, dtype=dtype, device=device), sampler_dim)
    return Memory(entries)


def panel(frame, model, seed, target, step, last_action, finished, success):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if frame.shape != (300, 400, 3) or frame.dtype != np.uint8:
        raise RuntimeError(f"Unexpected RoboTHOR RGB frame: {frame.shape} {frame.dtype}")
    canvas = Image.new("RGB", (400, 390), "white")
    canvas.paste(Image.fromarray(frame), (0, 60))
    draw = ImageDraw.Draw(canvas)
    color = "#1f77b4" if model == "gru1" else "#ff7f0e"
    draw.rectangle((0, 0, 399, 59), fill=color)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((9, 7), f"{model.upper()}  seed {seed}", fill="white", font=font)
    draw.text((9, 34), f"Find {target}", fill="white", font=small)
    status = "SUCCESS" if success else "FAILED" if finished else "RUNNING"
    draw.text((9, 365), f"step {step:03d} / 500   {last_action}   {status}",
              fill="#15202b", font=small)
    return np.asarray(canvas)


def rollout(run, episode, output, device, torch):
    import imageio
    import numpy as np
    from ai2thor.platform import CloudRendering
    from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
    from allenact.base_abstractions.sensor import SensorSuite
    from allenact_plugins.robothor_plugin.robothor_task_samplers import (
        ObjectNavDatasetTaskSampler,
    )
    from scripts.embclip.light5m_gru1_config import Light5MGRU1Config
    from scripts.embclip.light5m_gawf1_config import Light5MGaWF1Config

    model_name, seed = run["model"], run["seed"]
    config_cls = {"gru1": Light5MGRU1Config, "gawf1": Light5MGaWF1Config}[model_name]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    config = config_cls(headless=True, test_on_validation=True,
                        num_test_processes=1, test_gpu_ids=[0])
    if [sensor.uuid for sensor in config.SENSORS] != ["rgb_lowres", "goal_object_type_ind"]:
        raise RuntimeError("Policy sensor boundary changed")
    graph = SensorPreprocessorGraph(
        source_observation_spaces=SensorSuite(config.SENSORS).observation_spaces,
        preprocessors=config.preprocessors(),
    ).to(device)
    if set(graph.observation_spaces) != {"rgb_clip_resnet", "goal_object_type_ind"}:
        raise RuntimeError("Policy preprocessed observation boundary changed")
    policy = config.create_model(sensor_preprocessor_graph=graph).to(device)
    checkpoint = ROOT / run["checkpoint_path"]
    if sha256(checkpoint) != run["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    if int(state["total_steps"]) != 5_001_216:
        raise RuntimeError("Unexpected checkpoint training steps")
    policy.load_state_dict(state["model_state_dict"], strict=True)
    policy.eval()

    args = dict(
        scenes=[episode["scene"]], scene_directory=str(ROOT / "data/embclip-robothor/robothor-objectnav/val"),
        sensors=config.SENSORS, max_steps=config.MAX_STEPS, action_space=config.ACTION_SPACE,
        rewards_config=config.REWARD_CONFIG, seed=0, loop_dataset=False, allow_flipping=False,
        object_types=config.TARGET_TYPES,
        env_args={**config.env_args(), "gpu_device": 0, "platform": CloudRendering},
    )
    sampler = ObjectNavDatasetTaskSampler(**args)
    video_path = output / f"{model_name}_seed{seed}.mp4"
    actions = []
    try:
        sampler.episodes = {episode["scene"]: [episode]}
        sampler.scenes = [episode["scene"]]
        sampler.reset_tasks = sampler.max_tasks = 1
        sampler.reset()
        task = sampler.next_task()
        if task is None or task.task_info["id"] != episode["id"] or task.task_info["mirrored"]:
            raise RuntimeError("Sampler did not start the requested unmirrored episode")
        if config.MAX_STEPS != 500 or config.env_args()["renderDepthImage"]:
            raise RuntimeError("Unexpected visualization protocol")
        memory = initial_memory(policy, device, torch)
        previous = torch.zeros((1, 1), dtype=torch.long, device=device)
        mask = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
        with imageio.get_writer(str(video_path), fps=12, codec="libx264",
                                macro_block_size=2, quality=8) as writer:
            writer.append_data(panel(task.render("rgb"), model_name, seed,
                                     episode["object_type"], 0, "START", False, False))
            while not task.is_done():
                observation = task.get_observations()
                if set(observation) != {"rgb_lowres", "goal_object_type_ind"}:
                    raise RuntimeError("Unexpected policy observations")
                tensors = {key: torch.as_tensor(value, device=device).unsqueeze(0)
                           for key, value in observation.items()}
                with torch.no_grad():
                    processed = graph.get_observations(tensors)
                    policy_input = {key: value.unsqueeze(0) for key, value in processed.items()}
                    result, memory = policy(policy_input, memory, previous, mask)
                    action = int(result.distributions.sample().item())
                action_name = task.class_action_names()[action]
                task.step(action)
                actions.append(action_name)
                previous.fill_(action)
                mask.fill_(1.0)
                writer.append_data(panel(task.render("rgb"), model_name, seed,
                                         episode["object_type"], len(actions), action_name,
                                         task.is_done(), bool(task._success)))
                if len(actions) % 25 == 0:
                    print(f"{model_name} seed{seed}: {len(actions)} steps", flush=True)
        metrics = task.metrics()
        if not metrics or len(actions) > 500:
            raise RuntimeError("Rollout did not complete")
        receipt = dict(model=model_name, seed=seed, episode_id=episode["id"],
                       scene=episode["scene"], target=episode["object_type"],
                       action_sampling_seed=0, steps=len(actions), actions=actions,
                       success=bool(metrics["success"]), spl=float(metrics["spl"]),
                       checkpoint_sha256=run["checkpoint_sha256"],
                       policy_observations=["rgb_clip_resnet", "goal_object_type_ind"],
                       rgb_video=video_path.name, rgb_video_sha256=sha256(video_path))
        save_json(output / f"{model_name}_seed{seed}.json", receipt)
        return receipt
    finally:
        sampler.close()


def combine(first, second, output, episode):
    import imageio
    import numpy as np

    readers = [imageio.get_reader(str(output / item["rgb_video"])) for item in (first, second)]
    try:
        lengths = [reader.count_frames() for reader in readers]
        if lengths != [first["steps"] + 1, second["steps"] + 1]:
            raise RuntimeError(f"Frame/step count mismatch: {lengths}")
        last = [None, None]
        with imageio.get_writer(str(output / "side_by_side.mp4"), fps=12,
                                codec="libx264", macro_block_size=2, quality=8) as writer:
            for index in range(max(lengths)):
                for side, reader in enumerate(readers):
                    if index < lengths[side]:
                        last[side] = reader.get_data(index)
                writer.append_data(np.concatenate(last, axis=1))
    finally:
        for reader in readers:
            reader.close()
    video = output / "side_by_side.mp4"
    return dict(episode_id=episode["id"], scene=episode["scene"],
                target=episode["object_type"], fps=12, frames=max(lengths),
                video_path=str(video.relative_to(ROOT)), video_sha256=sha256(video),
                policies=[first, second], status="complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    identifier = args.experiment_id
    manifest_path = ROOT / "experiments/monitoring/jobs" / f"{identifier}.json"
    manifest = json.loads(manifest_path.read_text())
    output = ROOT / "results" / identifier
    if (manifest["experiment_id"] != identifier or manifest["status"] != "prepared"
            or manifest["mode"] != "qualitative_video" or output != ROOT / manifest["result_path"]
            or manifest["runner_sha256"] != sha256(Path(__file__))
            or len(manifest["policies"]) != 2
            or [(item["model"], item["seed"]) for item in manifest["policies"]]
            != [("gru1", 1), ("gawf1", 3)]):
        raise RuntimeError("Unexpected demonstration identity or source")
    contents = {path.name for path in output.iterdir()} if output.is_dir() else set()
    if (not output.is_dir() or output.is_symlink()
            or Path(__file__).resolve() != (output / "runner.py").resolve()
            or not {"runner.py"} <= contents
            or contents - {"runner.py", "run.log"}):
        raise RuntimeError("Result leaf must contain only this frozen runner and session log")
    dataset = ROOT / manifest["episode_dataset_path"]
    if sha256(dataset) != manifest["episode_dataset_sha256"]:
        raise RuntimeError("Validation episode file changed")
    episode = find_episode(dataset, manifest["episode_id"])
    if episode["scene"] != manifest["scene"] or episode["object_type"] != manifest["target"]:
        raise RuntimeError("Validation episode identity changed")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible SJC GPU")
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    manifest["status"] = "running"
    save_json(manifest_path, manifest)
    try:
        policies = [rollout(item, episode, output, device, torch) for item in manifest["policies"]]
        summary = combine(*policies, output, episode)
        save_json(output / "summary.json", summary)
        manifest["status"] = "complete"
        manifest["completion_evidence"] = {
            "summary_path": str((output / "summary.json").relative_to(ROOT)),
            "video_sha256": summary["video_sha256"], "return_code": 0,
            "actual_success": {item["model"]: item["success"] for item in policies},
        }
        save_json(manifest_path, manifest)
        print(json.dumps({"status": "complete", "video": summary["video_path"],
                          "frames": summary["frames"],
                          "actual_success": manifest["completion_evidence"]["actual_success"]}), flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = str(error)
        save_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
