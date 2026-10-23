"""Read-only protocol/data checks. Does not import Habitat, download, or train."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def flatten(value, prefix=""):
    result = {}
    for key, child in value.items():
        name = prefix + key
        if isinstance(child, dict):
            result.update(flatten(child, name + "."))
        else:
            result[name] = child
    return result


def check_config(root):
    config_dir = root / "configs/pointnav2020"
    protocol = json.loads((config_dir / "protocol.json").read_text())
    configs = {}
    for name, source_key, allowed_key in (
        ("task", "official_task_source", "allowed_task_overrides"),
        ("ddppo", "official_trainer_source", "allowed_trainer_overrides"),
    ):
        actual = yaml.safe_load((config_dir / (name + ".yaml")).read_text())
        original = yaml.safe_load((root / protocol[source_key]).read_text())
        a, b = flatten(actual), flatten(original)
        changed = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        forbidden = changed - set(protocol[allowed_key])
        if forbidden:
            raise ValueError("Changed official protocol: " + str(sorted(forbidden)))
        configs[name] = actual
    task, trainer = configs["task"], configs["ddppo"]
    if task["TASK"]["SENSORS"] != ["POINTGOAL_SENSOR"]:
        raise ValueError("Expected only the fixed initial pointgoal task sensor")
    if task["DATASET"] != {
        "TYPE": "PointNav-v1", "SPLIT": "train",
        "DATA_PATH": "data/datasets/pointnav/gibson/v2/{split}/{split}.json.gz",
        "SCENES_DIR": "data/scene_datasets/",
    }:
        raise ValueError("Expected Gibson v2 train episodes at the agreed project path")
    if trainer["BASE_TASK_CONFIG_PATH"] != "configs/pointnav2020/task.yaml":
        raise ValueError("Wrong base task config")
    for key in ("CHECKPOINT_FOLDER", "TENSORBOARD_DIR", "VIDEO_DIR", "EVAL_CKPT_PATH_DIR"):
        if "REQUIRES_EXPERIMENT_ID" not in trainer[key]:
            raise ValueError("Template output must require a registered experiment ID")
    return {"status": "pass", "protocol_id": protocol["protocol_id"],
            "policy_observations": protocol["policy_observations"],
            "online_gps_compass_pose": False,
            "official_model_and_optimizer_unchanged": True,
            "source_hashes": {protocol[k]: hashlib.sha256((root / protocol[k]).read_bytes()).hexdigest()
                              for k in ("official_task_source", "official_trainer_source")}}


def read_gzip(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def check_data(root):
    """Match every episode scene to an exact GLB; reject split overlap/empty data."""
    dataset_root = root / "data/datasets/pointnav/gibson/v2"
    scene_root = (root / "data/scene_datasets").resolve()
    errors, splits, scene_sets, missing_navmeshes = [], {}, {}, set()
    for split in ("train", "val"):
        folder = dataset_root / split
        index = folder / (split + ".json.gz")
        scene_ids, count = set(), 0
        if not index.is_file():
            errors.append("Missing " + str(index.relative_to(root)))
            continue
        metadata = read_gzip(index)
        pattern = metadata.get("content_scenes_path", "{data_path}/content/{scene}.json.gz")
        prefix, suffix = pattern.split("{scene}")
        content_dir = Path(prefix.format(data_path=str(folder)))
        if not content_dir.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Content path outside the declared split: " + pattern)
        files = [index]
        if content_dir.is_dir():
            files += sorted(p for p in content_dir.iterdir() if p.is_file() and p.name.endswith(suffix))
        for path in files:
            data = metadata if path == index else read_gzip(path)
            count += len(data["episodes"])
            # Scene paths repeat across episodes; resolve each distinct reference once.
            for raw in {episode["scene_id"] for episode in data["episodes"]}:
                if raw.startswith("data/scene_datasets/"):
                    raw = raw[len("data/scene_datasets/"):]
                scene = (scene_root / raw).resolve()
                if not scene.is_relative_to(scene_root / "gibson") or scene.suffix != ".glb":
                    raise ValueError("Unexpected Gibson scene reference: " + raw)
                scene_ids.add(str(scene.relative_to(scene_root)))
        for scene_id in scene_ids:
            scene = scene_root / scene_id
            if not scene.is_file() or scene.stat().st_size == 0:
                errors.append("Missing/empty scene: " + scene_id)
            if not scene.with_suffix(".navmesh").is_file():
                missing_navmeshes.add(scene_id)
        if count == 0:
            errors.append("Empty split: " + split)
        scene_sets[split] = scene_ids
        splits[split] = {"episodes": count, "scenes": sorted(scene_ids), "episode_files": len(files)}
    overlap = scene_sets.get("train", set()) & scene_sets.get("val", set())
    if overlap:
        errors.append("Train/val scene overlap: " + str(sorted(overlap)))
    return {"status": "fail" if errors else "pass", "splits": splits,
            "errors": errors, "missing_navmesh_files": sorted(missing_navmeshes),
            "note": "File coverage only. Simulator must load/recompute navmesh for the 2020 agent dimensions on a GPU worker before smoke acceptance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="store_true", help="Also check installed v2 train/val data")
    args = parser.parse_args()
    result = {"config": check_config(ROOT)}
    if args.data:
        result["data"] = check_data(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(args.data and result["data"]["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
