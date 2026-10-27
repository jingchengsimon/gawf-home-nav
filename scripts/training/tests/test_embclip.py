"""Offline regression checks for the unified seed0–4 launchers; no simulator."""

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.training.embclip import train


class TrainingChecks(unittest.TestCase):
    def training_manifest(self, seed, model="gru1"):
        runner = str(Path(train.__file__).relative_to(train.ROOT))
        return dict(experiment_id="training", host="sjc-remote", status="prepared",
                    model=dict(family=model), execution=dict(seed=seed,
                        physical_gpu_index=0, steps_budget=5_000_000,
                        checkpoint_interval_steps=100_000, global_train_envs=16),
                    source=dict(source_files={runner: train.sha256(Path(train.__file__))}))


    def test_independent_seeds_cannot_resume_another_run(self):
        for seed in range(1, 5):
            run = self.training_manifest(seed)
            run["resume_from"] = dict(checkpoint_path="other.pt")
            with self.subTest(seed=seed), self.assertRaisesRegex(RuntimeError, "independently"):
                train.validate_manifest(run, "training")
        run = self.training_manifest(0)
        run["resume_from"] = dict(checkpoint_path="pinned-seed0.pt")
        train.validate_manifest(run, "training")


    def test_invalid_protocols_and_sources_are_rejected(self):
        base = self.training_manifest(0)
        for section, key, value in (("execution", "seed", 5),
                                    ("execution", "steps_budget", 200_000_000),
                                    ("execution", "global_train_envs", 60)):
            run = copy.deepcopy(base)
            run[section][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                train.validate_manifest(run, "training")
        for status in ("complete", "paused"):
            run = copy.deepcopy(base)
            run["status"] = status
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                train.validate_manifest(run, "training")
        base["source"]["source_files"] = {}
        with self.assertRaisesRegex(RuntimeError, "pin"):
            train.validate_manifest(base, "training")


    def test_checkpoint_overshoot_seed_and_complete_state(self):
        for seed in range(5):
            state = dict(model_state_dict={}, optimizer_state_dict={},
                         training_pipeline_state_dict={}, scheduler_state={},
                         trainer_seed=seed, total_steps=5_001_216)
            fake_torch = SimpleNamespace(load=lambda *args, **kwargs: state)
            with patch.dict("sys.modules", torch=fake_torch):
                path = Path("exp__steps_000005001216.pt")
                self.assertEqual(train.verify_checkpoint(path, 5_000_000, seed), 5_001_216)
                with self.assertRaisesRegex(RuntimeError, "seed mismatch"):
                    train.verify_checkpoint(path, 5_000_000, seed + 1)
                state["total_steps"] = 5_004_096
                with self.assertRaisesRegex(RuntimeError, "beyond target"):
                    train.verify_checkpoint(Path("exp__steps_000005004096.pt"), 5_000_000, seed)
                del state["optimizer_state_dict"]
                with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                    train.verify_checkpoint(path, 5_000_000, seed)


    def test_all_five_seeds_share_training(self):
        for model in ("gru1", "gawf1"):
            for seed in range(5):
                train.validate_manifest(self.training_manifest(seed, model), "training")


if __name__ == "__main__":
    unittest.main()
