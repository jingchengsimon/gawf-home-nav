"""Offline regression checks for the unified seed0–4 launchers; no simulator."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.analysis.embclip import evaluate


class EvaluationChecks(unittest.TestCase):
    def evaluation_manifest(self, seed, model="gru1", gpu=0):
        runner = str(Path(evaluate.__file__).relative_to(evaluate.ROOT))
        digest = evaluate.sha256(Path(evaluate.__file__))
        return dict(experiment_id="evaluation", host="sjc-remote", mode="checkpoint_eval",
                    status="prepared", model=dict(family=model), seed=seed,
                    source_checkpoint={}, expected_episodes=1800, execution=dict(physical_gpu_index=gpu,
                        evaluation_seed=0, num_test_processes=15, test_on_validation=True),
                    runner_path=runner, runner_sha256=digest,
                    source=dict(source_files={runner: digest}))


    def test_evaluation_identity_and_source_guards(self):
        base = self.evaluation_manifest(4, "gawf1", 1)
        for section, key, value in ((None, "seed", 5), (None, "status", "partial"),
                                    (None, "experiment_id", "wrong"),
                                    (None, "runner_sha256", "old-runner"),
                                    ("execution", "evaluation_seed", 1),
                                    ("execution", "physical_gpu_index", 0),
                                    ("execution", "test_on_validation", False)):
            run = copy.deepcopy(base)
            (run[section] if section else run)[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                evaluate.validate_manifest(run, Path("evaluation.json"), 1)
        base["source"]["source_files"] = {}
        with self.assertRaisesRegex(RuntimeError, "pinned"):
            evaluate.validate_manifest(base, Path("evaluation.json"), 1)


    def test_official_reference_uses_the_same_metrics_finalizer(self):
        run = self.evaluation_manifest(0)
        run["source_checkpoint"] = dict(actual_training_steps=130_091_717, sha256="official")
        evaluate.validate_manifest(run, Path("evaluation.json"), 0)
        with tempfile.TemporaryDirectory() as folder, patch.object(evaluate, "ROOT", Path(folder)):
            out = Path(folder) / "results/evaluation"
            metrics = out / "allenact/metrics"
            metrics.mkdir(parents=True)
            (out / "exit.json").write_text('{"return_code": 0}')
            task = dict(task_info=dict(id="a", scene="scene"), success=True, spl=0.5)
            (metrics / "metrics__test_official.json").write_text(json.dumps([
                dict(training_steps=130_091_717, tasks=[task], success=1.0, spl=0.5, ep_length=20)]))
            run["units"] = [dict(status="running")]
            evaluate.finalize_metrics(Path(folder) / "manifest.json", run, out, {"a"}, {"scene"})
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["checkpoint_steps"], 130_091_717)
            self.assertEqual(summary["status"], "complete")


    def test_logged_evaluation_progress_and_failure(self):
        for return_code in (0, 2):
            process = SimpleNamespace(stdout=io.StringIO(
                "num_test_tasks_complete 10\nnum_test_tasks_complete 8\n"
                "num_test_tasks_complete 1790\nnum_test_tasks_complete 1801\n"),
                wait=lambda: return_code)
            with tempfile.TemporaryDirectory() as folder, patch.object(evaluate.subprocess, "Popen") as popen:
                popen.return_value.__enter__.return_value = process
                out = Path(folder)
                self.assertEqual(evaluate.run_logged_evaluation([], out, {}, 1800), return_code)
                progress = json.loads((out / "progress.json").read_text())
                self.assertEqual(progress["completed_episodes"], 1790)
                self.assertEqual(progress["status"], "failed" if return_code else "running")


    def test_finalization_preserves_partial_coverage_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(evaluate, "ROOT", Path(folder)):
            out = Path(folder) / "results/evaluation"
            metrics_dir = out / "allenact/metrics"
            metrics_dir.mkdir(parents=True)
            (out / "exit.json").write_text('{"return_code": 0}')
            metrics_file = metrics_dir / "metrics__test_fixture.json"
            task = dict(task_info=dict(id="episode-a", scene="val-scene"), success=True, spl=0.5)
            rows = [dict(training_steps=5_001_216, tasks=[task], success=1.0, spl=0.5, ep_length=20)]
            metrics_file.write_text(json.dumps(rows))
            run = dict(experiment_id="evaluation", expected_episodes=1800,
                       source_checkpoint=dict(sha256="checkpoint-hash"), units=[dict(status="running")])
            manifest_path = Path(folder) / "evaluation.json"
            evaluate.finalize_metrics(manifest_path, run, out, {"episode-a", "episode-b"}, {"val-scene"})
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["missing_episode_ids"], ["episode-b"])
            self.assertEqual(summary["success_rate"], 1.0)
            self.assertEqual(summary["spl"], 0.5)
            self.assertEqual(summary["checkpoint_sha256"], "checkpoint-hash")
            rows[0]["tasks"].append(task)
            metrics_file.write_text(json.dumps(rows))
            with self.assertRaisesRegex(RuntimeError, "identities"):
                evaluate.finalize_metrics(manifest_path, run, out, {"episode-a", "episode-b"}, {"val-scene"})


    def test_all_five_seeds_share_evaluation(self):
        for model in ("gru1", "gawf1"):
            for seed in range(5):
                evaluate.validate_manifest(self.evaluation_manifest(seed, model, 1), Path("evaluation.json"), 1)


if __name__ == "__main__":
    unittest.main()
