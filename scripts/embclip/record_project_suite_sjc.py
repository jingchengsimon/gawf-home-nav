"""SJC-only paired RoboTHOR RGB rollouts with evaluator-only top-down views.

Run a frozen copy at results/<experiment-id>/runner.py. The manifest freezes
checkpoints, episode files, rollout seeds and selection before any rollout.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor/embclip-allenact")]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(p)


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def episode(spec):
    p = ROOT / spec["dataset_path"]
    if sha(p) != spec["dataset_sha256"]:
        raise RuntimeError("Episode data hash mismatch")
    with gzip.open(p, "rt") as f:
        matches = [e for e in json.load(f) if e["id"] == spec["episode_id"]]
    if len(matches) != 1:
        raise RuntimeError("Episode ID must be unique")
    return matches[0]


def sampler_for(e, seed):
    from ai2thor.platform import CloudRendering
    from allenact_plugins.robothor_plugin.robothor_task_samplers import ObjectNavDatasetTaskSampler
    from scripts.embclip.light5m_gru1_config import Light5MGRU1Config
    c = Light5MGRU1Config(headless=True, test_on_validation=True,
                         num_test_processes=1, test_gpu_ids=[0])
    sampler = ObjectNavDatasetTaskSampler(
        scenes=[e["scene"]], scene_directory=str(ROOT / "data/embclip-robothor/robothor-objectnav/val"),
        sensors=c.SENSORS, max_steps=500, action_space=c.ACTION_SPACE,
        rewards_config=c.REWARD_CONFIG, seed=seed, loop_dataset=False,
        allow_flipping=False, object_types=c.TARGET_TYPES,
        env_args={**c.env_args(), "gpu_device": 0, "platform": CloudRendering},
    )
    sampler.episodes = {e["scene"]: [e]}
    sampler.scenes = [e["scene"]]
    sampler.reset_tasks = sampler.max_tasks = 1
    sampler.reset()
    task = sampler.next_task()
    if task is None or task.task_info["id"] != e["id"] or task.task_info["mirrored"]:
        sampler.close()
        raise RuntimeError("Requested episode was not initialized")
    return sampler, task


def pose(task):
    a = task.env.last_event.metadata["agent"]
    return {"position": dict(a["position"]), "yaw": a["rotation"]["y"],
            "horizon": a["cameraHorizon"]}


def checked_action(controller, **kwargs):
    event = controller.step(**kwargs)
    if not event.metadata["lastActionSuccess"]:
        raise RuntimeError(str(kwargs) + ": " + event.metadata.get("errorMessage", ""))
    return event


def map_view(task):
    """Capture a static evaluator view, then restore first-person rendering."""
    import numpy as np
    controller = task.env.controller
    before = pose(task)
    event = checked_action(controller, action="ToggleMapView")
    frame = event.frame.copy()
    camera = dict(event.metadata["cameraPosition"])
    orth = float(event.metadata["cameraOrthSize"])
    checked_action(controller, action="ToggleMapView")
    after = pose(task)
    if max(abs(before["position"][k] - after["position"][k]) for k in ("x", "y", "z")) > 1e-5:
        raise RuntimeError("Map capture changed agent position")
    if abs(before["yaw"] - after["yaw"]) > 1e-4:
        raise RuntimeError("Map capture changed agent yaw")
    if not isinstance(frame, np.ndarray) or frame.shape != (300, 400, 3):
        raise RuntimeError("Invalid top-down frame")
    targets = [dict(o["position"]) for o in task.env.last_event.metadata["objects"]
               if o["objectType"] == task.task_info["object_type"]]
    return frame, {"camera": camera, "orthographic_size": orth,
                   "width": 400, "height": 300, "target_positions": targets}


def load_bundle(run):
    import torch
    from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
    from allenact.base_abstractions.sensor import SensorSuite
    from scripts.embclip.light5m_gru1_config import Light5MGRU1Config
    from scripts.embclip.light5m_gawf1_config import Light5MGaWF1Config
    cls = {"gru1": Light5MGRU1Config, "gawf1": Light5MGaWF1Config}[run["model"]]
    c = cls(headless=True, test_on_validation=True, num_test_processes=1, test_gpu_ids=[0])
    if [s.uuid for s in c.SENSORS] != ["rgb_lowres", "goal_object_type_ind"]:
        raise RuntimeError("Policy sensor boundary changed")
    graph = SensorPreprocessorGraph(
        source_observation_spaces=SensorSuite(c.SENSORS).observation_spaces,
        preprocessors=c.preprocessors(),
    ).to(torch.device("cuda:0"))
    if set(graph.observation_spaces) != {"rgb_clip_resnet", "goal_object_type_ind"}:
        raise RuntimeError("Policy observation boundary changed")
    model = c.create_model(sensor_preprocessor_graph=graph).cuda().eval()
    checkpoint = ROOT / run["checkpoint_path"]
    if sha(checkpoint) != run["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash mismatch")
    state = torch.load(checkpoint, map_location="cpu")
    if int(state["total_steps"]) != 5_001_216:
        raise RuntimeError("Unexpected training budget")
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model, graph


def memory_for(model):
    import torch
    from allenact.base_abstractions.misc import Memory
    entries = {}
    for name, (dims, dtype) in model.recurrent_memory_specification.items():
        shape = tuple(1 if size is None else size for _, size in dims)
        entries[name] = (torch.zeros(shape, dtype=dtype, device="cuda:0"),
                         [d[0] for d in dims].index("sampler"))
    return Memory(entries)


def canvas(rgb, topdown, projection, trajectory, run, e, seed, action, done, success):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1000, 470
    out = Image.new("RGB", (width, height), "white")
    color = "#1f77b4" if run["model"] == "gru1" else "#ff7f0e"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = small = ImageFont.load_default()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, width, 58), fill=color)
    draw.text((14, 8), f'{run["model"].upper()}  training seed {run["seed"]}  |  Find {e["object_type"]}', fill="white", font=font)
    draw.text((14, 35), f'{e["id"]}  |  rollout seed {seed}', fill="white", font=small)
    map_img = Image.fromarray(topdown).resize((500, 375))
    out.paste(map_img, (0, 70))
    out.paste(Image.fromarray(rgb).resize((500, 375)), (500, 70))
    draw = ImageDraw.Draw(out)
    cam = projection["camera"]
    orth = projection["orthographic_size"]
    # Unity orthographicSize is the vertical half-span; aspect ratio is 4:3.
    def xy(position):
        return ((position["x"] - cam["x"]) / (2 * orth * 4 / 3) * 500 + 250,
                70 + (cam["z"] - position["z"]) / (2 * orth) * 375 + 187.5)
    points = [xy(p["position"]) for p in trajectory]
    for target in projection.get("target_positions", []):
        tx, ty = xy(target)
        draw.rectangle((tx-6, ty-6, tx+6, ty+6), outline="#d02ce0", width=3)
    if len(points) > 1:
        draw.line(points, fill=color, width=4)
    x0, y0 = points[0]
    draw.ellipse((x0 - 5, y0 - 5, x0 + 5, y0 + 5), fill="white", outline="#111111", width=2)
    x, y = points[-1]
    yaw = math.radians(trajectory[-1]["yaw"])
    tip = (x + 12 * math.sin(yaw), y - 12 * math.cos(yaw))
    left = (x + 7 * math.sin(yaw + 2.4), y - 7 * math.cos(yaw + 2.4))
    right = (x + 7 * math.sin(yaw - 2.4), y - 7 * math.cos(yaw - 2.4))
    draw.polygon((tip, left, right), fill=color, outline="white")
    status = "SUCCESS" if success else "FAILED" if done else "RUNNING"
    draw.text((14, 449), f'Top-down trajectory  |  step {len(trajectory)-1:03d}/500  |  {action}  |  {status}', fill="#15202b", font=small)
    draw.text((750, 449), "Agent RGB", fill="#15202b", font=small)
    return np.asarray(out)


def record(run, e, seed, output, bundle):
    import imageio
    import numpy as np
    import torch
    from PIL import Image
    model, graph = bundle
    set_seed(seed)
    sampler, task = sampler_for(e, seed)
    prefix = run["model"] + "_rollout" + str(seed)
    actions, trajectory, observations = [], [], []
    try:
        topdown, projection = map_view(task)
        Image.fromarray(topdown).save(output / (prefix + "_map.png"))
        checked_action(task.env.controller, action="SetRandomSeed", seed=1000 + seed)
        start = pose(task)
        if max(abs(start["position"][k] - e["initial_position"][k]) for k in ("x", "y", "z")) > 1e-3:
            raise RuntimeError("Episode start position mismatch")
        if abs((start["yaw"]-e["initial_orientation"]+180)%360-180) > 1e-3:
            raise RuntimeError("Episode start yaw mismatch")
        if abs(start["horizon"]-e["initial_horizon"]) > 1e-3:
            raise RuntimeError("Episode camera horizon mismatch")
        first_rgb = task.render("rgb").copy()
        Image.fromarray(first_rgb).save(output / (prefix + "_start.png"))
        trajectory.append(start)
        mem = memory_for(model)
        prev = torch.zeros((1, 1), dtype=torch.long, device="cuda:0")
        mask = torch.zeros((1, 1, 1), device="cuda:0")
        generator = torch.Generator(device="cuda:0").manual_seed(seed)
        video = output / (prefix + ".mp4")
        with imageio.get_writer(str(video), fps=12, codec="libx264", quality=8, macro_block_size=2) as writer:
            writer.append_data(canvas(first_rgb, topdown, projection, trajectory, run, e, seed, "START", False, False))
            while not task.is_done():
                obs = task.get_observations()
                if set(obs) != {"rgb_lowres", "goal_object_type_ind"}:
                    raise RuntimeError("Unexpected actor inputs")
                if run["model"] == "gru1" and seed == 0 and len(actions) < 33:
                    observations.append({k: np.array(v) for k, v in obs.items()})
                tensors = {k: torch.as_tensor(v, device="cuda:0").unsqueeze(0) for k, v in obs.items()}
                with torch.no_grad():
                    processed = graph.get_observations(tensors)
                    result, mem = model({k: v.unsqueeze(0) for k, v in processed.items()}, mem, prev, mask)
                    action = int(torch.multinomial(result.distributions.probs.reshape(-1), 1, generator=generator).item())
                name = task.class_action_names()[action]
                task.step(action)
                actions.append({"index": action, "name": name})
                prev.fill_(action)
                mask.fill_(1)
                trajectory.append(pose(task))
                writer.append_data(canvas(task.render("rgb"), topdown, projection, trajectory, run, e, seed,
                                          name, task.is_done(), bool(task._success)))
                if len(actions) % 100 == 0:
                    print(e["id"], prefix, len(actions), flush=True)
        if observations:
            np.savez_compressed(output / "shared_history.npz",
                                rgb=np.stack([o["rgb_lowres"] for o in observations]),
                                goal=np.stack([o["goal_object_type_ind"] for o in observations]),
                                actions=np.array([a["index"] for a in actions[:len(observations)]], dtype=np.int64))
        metric = task.metrics()
        receipt = {"model": run["model"], "training_seed": run["seed"],
                   "checkpoint_sha256": run["checkpoint_sha256"], "episode": e,
                   "action_sampling_seed": seed, "simulator_seed": 1000 + seed,
                   "simulator_seed_action": "SetRandomSeed", "start_actual": start,
                   "start_rgb_sha256": sha(output / (prefix + "_start.png")),
                   "actions": actions, "trajectory": trajectory,
                   "map_projection": projection, "steps": len(actions),
                   "success": bool(metric["success"]), "spl": float(metric["spl"]),
                   "video": str(video.relative_to(ROOT)), "video_sha256": sha(video),
                   "actor_inputs": ["rgb_clip_resnet", "goal_object_type_ind", "recurrent_state"],
                   "evaluator_only": ["top_down_image", "agent_pose", "shortest_path", "metrics"]}
        write(output / (prefix + ".json"), receipt)
        print("FINISHED", e["id"], prefix, receipt["steps"], receipt["success"], flush=True)
        return receipt
    finally:
        sampler.close()


def paired_video(a, b, out):
    import imageio
    import numpy as np
    from PIL import Image
    if a["episode"] != b["episode"] or a["start_actual"] != b["start_actual"]:
        raise RuntimeError("Pair does not share the same initial task")
    starts = [np.asarray(Image.open(out.parent / (r["model"] + "_rollout" + str(r["action_sampling_seed"]) + "_start.png"))).astype(float) for r in (a,b)]
    difference = np.abs(starts[0]-starts[1])
    initial_rgb_mae = float(difference.mean())
    changed_fraction = float((difference.max(2)>20).mean())
    # Independent Unity renderers have small antialiasing / shadow differences.
    # Require identical actual pose and reject substantial visual discrepancies.
    if initial_rgb_mae > 1.0 or changed_fraction > 0.01:
        raise RuntimeError("Pair initial rendered scenes differ substantially")
    write(out.with_suffix(".fairness.json"), {"same_episode": True, "same_actual_pose": True,
          "same_initial_rgb_sha256": a["start_rgb_sha256"] == b["start_rgb_sha256"],
          "initial_rgb_mae_255": initial_rgb_mae, "fraction_pixels_difference_over_20": changed_fraction,
          "thresholds": {"mae_255": 1.0, "fraction_pixels_over_20": 0.01},
          "policy_seed": a["action_sampling_seed"], "simulator_seed": a["simulator_seed"]})
    readers = [imageio.get_reader(str(ROOT / r["video"])) for r in (a, b)]
    lengths = [r.count_frames() for r in readers]
    if lengths != [a["steps"] + 1, b["steps"] + 1]:
        raise RuntimeError("Video frames do not match action steps")
    try:
        last = [None, None]
        with imageio.get_writer(str(out), fps=12, codec="libx264", quality=8, macro_block_size=2) as writer:
            for i in range(max(lengths)):
                for j, reader in enumerate(readers):
                    if i < lengths[j]:
                        last[j] = reader.get_data(i)
                # Model panels remain independent: map+RGB on each row.
                writer.append_data(np.concatenate(last, axis=0))
                if i == min(24, max(lengths)-1):
                    Image.fromarray(np.concatenate(last, axis=0)).save(out.with_suffix(".png"))
    finally:
        for r in readers:
            r.close()


def smoke(e, output):
    from PIL import Image
    traces = []
    for attempt in range(2):
        sampler, task = sampler_for(e, 0)
        try:
            frame, projection = map_view(task)
            Image.fromarray(frame).save(output / ("map_probe" + str(attempt) + ".png"))
            checked_action(task.env.controller, action="SetRandomSeed", seed=1000)
            rows = [pose(task)]
            for action in ("MoveAhead", "RotateRight", "MoveAhead", "RotateLeft"):
                task.env.controller.step(action=action)
                rows.append(pose(task))
            traces.append(rows)
        finally:
            sampler.close()
    delta = max(abs(a["position"][k] - b["position"][k]) for a,b in zip(*traces) for k in ("x","y","z"))
    yaw_delta = max(abs(a["yaw"]-b["yaw"]) for a,b in zip(*traces))
    write(output / "smoke.json", {"episode": e, "traces": traces, "projection": projection,
                                  "max_position_delta": delta, "max_yaw_delta": yaw_delta,
                                  "repeat_probe_pass": delta < 1e-5 and yaw_delta < 1e-4})
    if delta >= 1e-5 or yaw_delta >= 1e-4:
        raise RuntimeError("Seeded simulator repeat probe was not reproducible")


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    mp = ROOT / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    m = json.loads(mp.read_text())
    out = ROOT / m["result_path"]
    if m["status"] != "prepared" or m["runner_sha256"] != sha(__file__) or out != Path(__file__).resolve().parent:
        raise RuntimeError("Unexpected source or result identity")
    if {p.name for p in out.iterdir()} - {"runner.py", "run.log"}:
        raise RuntimeError("Result leaf is not fresh")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible SJC GPU")
    torch.set_num_threads(1)
    m["status"] = "running"
    write(mp, m)
    try:
        if args.smoke:
            smoke(episode(m["episodes"][0]), out)
            summary = {"status": "complete", "smoke": True}
        else:
            bundles = {r["model"]: load_bundle(r) for r in m["policies"]}
            rows = []
            for spec in m["episodes"]:
                e = episode(spec)
                leaf = out / e["id"]
                leaf.mkdir()
                for seed in m["rollout_seeds"]:
                    source = ROOT / m.get("reuse_completed_from", "__no_source__") / e["id"]
                    old_receipts = [source / (r["model"] + "_rollout" + str(seed) + ".json") for r in m["policies"]]
                    if all(p.is_file() for p in old_receipts):
                        pair = []
                        for run, path in zip(m["policies"], old_receipts):
                            receipt = json.loads(path.read_text())
                            video_source = ROOT / receipt["video"]
                            if (receipt["episode"] != e or receipt["checkpoint_sha256"] != run["checkpoint_sha256"]
                                    or sha(video_source) != receipt["video_sha256"] or receipt["action_sampling_seed"] != seed):
                                raise RuntimeError("Invalid previously completed rollout")
                            prefix = run["model"] + "_rollout" + str(seed)
                            for suffix in (".mp4", "_map.png", "_start.png"):
                                shutil.copy2(source / (prefix + suffix), leaf / (prefix + suffix))
                            if (source / "shared_history.npz").is_file() and not (leaf / "shared_history.npz").exists():
                                shutil.copy2(source / "shared_history.npz", leaf / "shared_history.npz")
                            receipt["reused_from"] = str(path.relative_to(ROOT))
                            receipt["video"] = str((leaf / video_source.name).relative_to(ROOT))
                            write(leaf / path.name, receipt)
                            pair.append(receipt)
                    else:
                        pair = [record(r, e, seed, leaf, bundles[r["model"]]) for r in m["policies"]]
                    video = leaf / ("paired_rollout" + str(seed) + ".mp4")
                    paired_video(*pair, video)
                    rows.append({"episode_id": e["id"], "scene": e["scene"], "target": e["object_type"],
                                 "rollout_seed": seed, "models": [{k:r[k] for k in ("model","training_seed","steps","success","spl","video","video_sha256")} for r in pair],
                                 "paired_video": str(video.relative_to(ROOT)), "paired_video_sha256": sha(video)})
                    write(out / "progress.json", {"completed_pairs": len(rows), "expected_pairs": len(m["episodes"])*len(m["rollout_seeds"])})
            summary = {"status": "complete", "pairs": rows, "expected_pairs": len(m["episodes"])*len(m["rollout_seeds"]),
                       "trainable_parameters": {k:sum(p.numel() for p in b[0].parameters() if p.requires_grad) for k,b in bundles.items()}}
        write(out / "summary.json", summary)
        m["status"] = "complete"
        for unit in m.get("units", []):
            unit["status"] = "complete"
        m["completion_evidence"] = {"return_code": 0, "summary_sha256": sha(out / "summary.json")}
        write(mp, m)
    except Exception as error:
        m["status"] = "failed"
        m["failure"] = repr(error)
        write(mp, m)
        raise


if __name__ == "__main__":
    main()
