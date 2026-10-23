"""Small checks for forbidden observations and incomplete/mis-split data."""

import gzip
import json
from pathlib import Path
import tempfile
import shutil
import unittest

import yaml

from check_pointnav2020 import ROOT, check_config, check_data


class ProtocolChecks(unittest.TestCase):
    def test_new_online_sensor_is_rejected(self):
        self.assertEqual(check_config(ROOT)["status"], "pass")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("configs", "vendor"):
                if relative == "vendor":
                    (root / relative).symlink_to(ROOT / relative, target_is_directory=True)
                else:
                    shutil.copytree(ROOT / relative, root / relative)
            path = root / "configs/pointnav2020/task.yaml"
            task = yaml.safe_load(path.read_text())
            task["TASK"]["SENSORS"].append("GPS_SENSOR")
            path.write_text(yaml.safe_dump(task))
            with self.assertRaises(ValueError):
                check_config(root)

    def test_missing_scene_and_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, scene in (("train", "a"), ("val", "b")):
                folder = root / "data/datasets/pointnav/gibson/v2" / split
                folder.mkdir(parents=True)
                with gzip.open(folder / (split + ".json.gz"), "wt") as stream:
                    json.dump({"episodes": [{"scene_id": "data/scene_datasets/gibson/" + scene + ".glb"}]}, stream)
            self.assertEqual(check_data(root)["status"], "fail")
            scenes = root / "data/scene_datasets/gibson"
            scenes.mkdir(parents=True)
            for name in ("a", "b"):
                (scenes / (name + ".glb")).write_bytes(b"fixture")
            self.assertEqual(check_data(root)["status"], "pass")
            with gzip.open(root / "data/datasets/pointnav/gibson/v2/val/val.json.gz", "wt") as stream:
                json.dump({"episodes": [{"scene_id": "data/scene_datasets/gibson/a.glb"}]}, stream)
            self.assertIn("overlap", check_data(root)["errors"][0])


if __name__ == "__main__":
    unittest.main()
