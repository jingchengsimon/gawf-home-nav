"""Small offline checks for the dataset trust boundary."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.training.embclip import prepare_assets as assets


class AssetChecks(unittest.TestCase):
    def test_archive_paths(self):
        for name in ('/etc/passwd', 'root/../../escape'):
            with self.assertRaises(AssertionError):
                assets.stripped_member(name)
        self.assertEqual(assets.stripped_member('root/train/episodes'), Path('train/episodes'))

    def test_valid_episodes_and_corrupt_gzip(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(assets, 'DATA', Path(folder)):
            for split, scene in [('train', 'FloorPlan_Train1_1'), ('val', 'FloorPlan_Val1_1')]:
                path = assets.DATA / split / 'episodes' / (scene + '.json.gz')
                path.parent.mkdir(parents=True)
                with gzip.open(path, 'wt') as stream:
                    json.dump([dict(id=str(i), object_type=category,
                                    initial_position=dict(x=0, y=0, z=0), initial_orientation=0)
                               for i, category in enumerate(sorted(assets.TARGETS))], stream)
            self.assertEqual(assets.verify_episodes()['splits']['val']['episodes'], 12)
            damaged = bytearray(path.read_bytes())
            damaged[-8] ^= 1
            path.write_bytes(damaged)
            with self.assertRaises(gzip.BadGzipFile):
                assets.verify_episodes()


if __name__ == '__main__':
    unittest.main()
