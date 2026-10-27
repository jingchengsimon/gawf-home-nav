"""Label-free SJC RoboTHOR rollouts, with successful terminal observation probes.

Run a frozen copy at results/<experiment-id>/runner.py. The manifest freezes
checkpoints, episode files, rollout seeds and selection before any rollout.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import os
import shlex
import subprocess
from datetime import datetime
import sys

ROOT = (Path(__file__).resolve().parents[3] if Path(__file__).resolve().parent.name == "embclip"
        else Path(__file__).resolve().parents[2])
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
    from scripts.training.embclip.config import EmbCLIPConfig
    c = EmbCLIPConfig(headless=True, test_on_validation=True,
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
    from scripts.training.embclip.config import EmbCLIPConfig
    c = EmbCLIPConfig(model=run["model"], headless=True, test_on_validation=True, num_test_processes=1, test_gpu_ids=[0])
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
    if int(state["total_steps"]) != 5_001_216 or int(state["trainer_seed"]) != run["seed"]:
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
    from PIL import Image, ImageDraw
    out = Image.new("RGB", (1000, 376), "white")
    out.paste(Image.fromarray(topdown).resize((500,375)), (0,0))
    out.paste(Image.fromarray(rgb).resize((500,375)), (500,0))
    draw = ImageDraw.Draw(out)
    color = "#1f77b4" if run["model"] == "gru1" else "#ff7f0e"
    cam, orth = projection["camera"], projection["orthographic_size"]
    def xy(p):
        return ((p["x"]-cam["x"])/(2*orth*4/3)*500+250,
                (cam["z"]-p["z"])/(2*orth)*375+187.5)
    for p in projection.get("target_positions",[]):
        x,y=xy(p);draw.rectangle((x-6,y-6,x+6,y+6),outline="#d02ce0",width=3)
    points=[xy(p["position"]) for p in trajectory]
    if len(points)>1:draw.line(points,fill=color,width=4)
    x,y=points[0];draw.ellipse((x-5,y-5,x+5,y+5),fill="white",outline="black",width=2)
    x,y=points[-1];yaw=math.radians(trajectory[-1]["yaw"])
    draw.polygon(((x+12*math.sin(yaw),y-12*math.cos(yaw)),
                  (x+7*math.sin(yaw+2.4),y-7*math.cos(yaw+2.4)),
                  (x+7*math.sin(yaw-2.4),y-7*math.cos(yaw-2.4))),fill=color,outline="white")
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
                before_memory = {name: mem.tensor(name).detach().cpu().clone()
                                 for name in model.recurrent_memory_specification}
                before_prev, before_mask = prev.detach().cpu().clone(), mask.detach().cpu().clone()
                with torch.no_grad():
                    processed = graph.get_observations(tensors)
                    result, mem = model({k: v.unsqueeze(0) for k, v in processed.items()}, mem, prev, mask)
                    action = int(torch.multinomial(result.distributions.probs.reshape(-1), 1, generator=generator).item())
                name = task.class_action_names()[action]
                task.step(action)
                if task.is_done():
                    torch.save(dict(rgb=np.array(obs["rgb_lowres"]),goal=np.array(obs["goal_object_type_ind"]),
                        memory=before_memory,prev_actions=before_prev,masks=before_mask,
                        action=action,success=bool(task._success),decision_step=len(actions),
                        target_visible=any(o["objectType"]==e["object_type"] and o["visible"]
                                           for o in task.env.last_event.metadata["objects"])),
                        output/(prefix+"_terminal_probe.pt"))
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
                   "terminal_probe_sha256": sha(output / (prefix + "_terminal_probe.pt")),
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


def record_main():
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


def replay_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-id', required=True)
    args = p.parse_args()
    mp = ROOT / 'experiments/monitoring/jobs' / (args.experiment_id + '.json')
    manifest = json.loads(mp.read_text())
    out = ROOT / manifest['result_path']
    if (manifest['status'] != 'prepared' or sha(__file__) != manifest['runner_sha256']
            or out != Path(__file__).resolve().parent
            or {p.name for p in out.iterdir()} - {'runner.py', 'run.log'}):
        raise RuntimeError('Manifest, source, or fresh output leaf mismatch')
    import imageio
    from PIL import Image
    manifest['status'] = 'running'
    write(mp, manifest)
    rows = []
    try:
        for spec in manifest['episodes']:
            e = episode(spec)
            leaf = out / e['id']
            leaf.mkdir()
            pair = []
            for run in manifest['policies']:
                checkpoint = ROOT / run['checkpoint_path']
                metrics_path = ROOT / run['evaluation_metrics_path']
                if sha(checkpoint) != run['checkpoint_sha256'] or sha(metrics_path) != run['evaluation_metrics_sha256']:
                    raise RuntimeError('Checkpoint or source metrics hash mismatch')
                metrics = json.loads(metrics_path.read_text())
                if len(metrics) != 1 or metrics[0]['training_steps'] != 5001216:
                    raise RuntimeError('Source training budget mismatch')
                task = next(t for t in metrics[0]['tasks'] if t['task_info']['id'] == e['id'])
                info = task['task_info']
                if info['mirrored'] or info['materials_randomized'] or not task['success']:
                    raise RuntimeError('Require original, successful validation episode')
                path, names = info['followed_path'], info['taken_actions']
                # AllenAct records End without appending a pose: End does not move.
                if len(path) == len(names) and names[-1] == 'End':
                    path = path + [path[-1]]
                if len(path) != len(names) + 1 or len(names) != task['ep_length']:
                    raise RuntimeError('Recorded pose/action count mismatch')
                sampler, env_task = sampler_for(e, 0)
                prefix = run['model'] + '_rollout0'
                trajectory = []
                video = leaf / (prefix + '.mp4')
                try:
                    topdown, projection = map_view(env_task)
                    Image.fromarray(topdown).save(leaf / (prefix + '_map.png'))
                    with imageio.get_writer(str(video), fps=12, codec='libx264', quality=8, macro_block_size=2) as writer:
                        for i, point in enumerate(path):
                            event = env_task.env.controller.step(action='TeleportFull',
                                x=point['x'], y=point['y'], z=point['z'],
                                rotation=point['rotation'], horizon=point['horizon'],
                                forceAction=True)
                            if not event.metadata['lastActionSuccess']:
                                raise RuntimeError(event.metadata.get('errorMessage'))
                            actual = pose(env_task)
                            if max(abs(actual['position'][k] - point[k]) for k in ('x', 'y', 'z')) > 1e-3:
                                raise RuntimeError('Replay position differs from logged pose')
                            if abs((actual['yaw'] - point['rotation']['y'] + 180) % 360 - 180) > 1e-3 or abs(actual['horizon'] - point['horizon']) > 1e-3:
                                raise RuntimeError('Replay view differs from logged orientation')
                            trajectory.append(actual)
                            rgb = env_task.render('rgb').copy()
                            if i == 0:
                                Image.fromarray(rgb).save(leaf / (prefix + '_start.png'))
                            writer.append_data(canvas(rgb, topdown, projection, trajectory, run, e, 0,
                                names[i-1] if i else 'START', i == len(path)-1, bool(task['success'])))
                        Image.fromarray(rgb).save(leaf / (prefix + '_terminal.png'))
                    visible = any(o['objectType'] == e['object_type'] and o['visible'] for o in event.metadata['objects'])
                    if not visible:
                        raise RuntimeError('Target not visible in reconstructed successful terminal frame')
                    receipt = dict(model=run['model'], training_seed=run['seed'],
                        checkpoint_sha256=run['checkpoint_sha256'], episode=e,
                        action_sampling_seed=0, simulator_seed=None,
                        seed_scope='original AllenAct evaluation seed0; original per-action RNG states were not saved',
                        start_actual=trajectory[0], trajectory=trajectory,
                        actions=[dict(index=info['action_names'].index(n), name=n) for n in names],
                        steps=len(names), success=bool(task['success']), spl=float(task['spl']),
                        video=str(video.relative_to(ROOT)), video_sha256=sha(video),
                        start_rgb_sha256=sha(leaf / (prefix + '_start.png')),
                        rendering_mode='logged_validation_pose_replay',
                        source_metrics=str(metrics_path.relative_to(ROOT)), source_metrics_sha256=sha(metrics_path),
                        terminal_target_visible=visible, map_projection=projection,
                        limits='Images are re-rendered at logged poses, not original pixel-identical observations. No policy is executed during replay; metrics and actions are unchanged source evaluation records.')
                    write(leaf / (prefix + '.json'), receipt)
                    pair.append(receipt)
                finally:
                    sampler.close()
            video = leaf / 'paired_rollout0.mp4'
            paired_video(*pair, video)
            rows.append(dict(episode_id=e['id'], target=e['object_type'], models=pair))
        summary = dict(status='complete', pairs=rows, rendering_mode='logged_validation_pose_replay')
        write(out / 'summary.json', summary)
        manifest['status'] = manifest['units'][0]['status'] = 'complete'
        manifest['completion_evidence'] = dict(return_code=0, summary_sha256=sha(out / 'summary.json'))
        write(mp, manifest)
    except BaseException as error:
        manifest['status'] = manifest['units'][0]['status'] = 'failed'
        manifest['failure'] = repr(error)
        write(mp, manifest)
        raise



def launch_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    operation = "record"
    root = args.root.resolve()
    mp = root / "experiments/monitoring/jobs" / (args.experiment_id + ".json")
    m = json.loads(mp.read_text())
    leaf = root / m["result_path"]
    operation = m["execution"]["operation"]
    if operation not in {"record", "replay", "gradcam-replay", "gradcam-sample"}:
        raise RuntimeError("Unknown frozen media operation")
    inputs = m.get("input_files", {})
    if any(Path(name).name != name for name in inputs):
        raise RuntimeError("Input files must be leaf filenames")
    if m["status"] != "prepared" or {p.name for p in leaf.iterdir()} != {"runner.py", *inputs}:
        raise RuntimeError("Job or result leaf is not fresh")
    for name, digest in inputs.items():
        if hashlib.sha256((leaf / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Input file hash mismatch: " + name)
    if hashlib.sha256((leaf / "runner.py").read_bytes()).hexdigest() != m["runner_sha256"]:
        raise RuntimeError("Frozen runner hash mismatch")
    for relative, digest in m.get("source_files", {}).items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source changed: " + relative)
    if not args.worker:
        session = m["execution"]["session"]
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            raise RuntimeError("Session already exists")
        command = ["python3", str(Path(__file__).resolve()), "launch", "--root", str(root),
                   "--experiment-id", args.experiment_id, "--worker"]
        subprocess.run(["tmux", "new-session", "-d", "-s", session, shlex.join(command)], check=True)
        print("STARTED", session, flush=True)
        return
    gpu = int(m["execution"]["physical_gpu_index"])
    uuid, free = subprocess.check_output(["nvidia-smi", "--id=" + str(gpu),
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader"], text=True)
    if int(free) < 11000 or any(line.split(",")[0].strip() == uuid.strip() for line in apps.splitlines()):
        raise RuntimeError("Target GPU is occupied")
    if shutil.disk_usage(leaf).free < 5 * 1024**3:
        raise RuntimeError("Insufficient result space")
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), HOME=str(root / "assets/embclip-robothor/home"),
               PYTHONPATH=str(root / "vendor/embclip-allenact") + ":" + str(root),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PATH=str(root / "assets/embclip-robothor/vulkan-tools/package/usr/bin") + ":" + env["PATH"])
    env.pop("DISPLAY", None)
    with (leaf / "run.log").open("x") as log:
        completed = subprocess.run([str(root / ".envs/embclip-robothor/bin/python"), "-u",
                                   str(leaf / "runner.py"), operation, "--experiment-id", args.experiment_id],
                                  cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(return_code=completed.returncode, finished_at=datetime.now().astimezone().isoformat())
    (leaf / "exit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if completed.returncode:
        m = json.loads(mp.read_text())
        m["status"] = "failed"
        for unit in m.get("units", []):
            unit["status"] = "failed"
        mp.write_text(json.dumps(m, indent=2) + "\n")
    subprocess.run(["python3", "-m", "scripts.training.monitor", "--rebuild"], cwd=root, check=True)
    raise SystemExit(completed.returncode)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=('record', 'replay', 'launch'))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    {'record': record_main, 'replay': replay_main, 'launch': launch_main}[args.operation]()


if __name__ == "__main__":
    main()
