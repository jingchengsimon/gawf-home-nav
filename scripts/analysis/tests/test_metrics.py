"""Analysis regression checks for campaign coverage and complete comparison execution."""
import contextlib
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.analysis.embclip import metrics
from scripts.analysis.embclip.metrics import campaign_runs


class AnalysisChecks(unittest.TestCase):
    def test_five_seed_campaign_does_not_depend_on_date_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = root / 'experiments/monitoring/jobs'
            jobs.mkdir(parents=True)
            for model in ('gru1', 'gawf1'):
                for seed in range(5):
                    eid = f'nav-robothor-embclip-light-{model}-eval-val1800-sjc-v1-seed{seed}-20990101'
                    (jobs / (eid + '.json')).write_text(json.dumps(dict(
                        experiment_id=eid, host='sjc-remote', model=dict(family=model), seed=seed)))
            self.assertEqual(len(campaign_runs(root, 'eval-val1800')), 10)
            sample = sorted(jobs.glob('*.json'))[0]
            sample.unlink()
            with self.assertRaisesRegex(RuntimeError, 'Missing'):
                campaign_runs(root, 'eval-val1800')


    def test_duplicate_campaign_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = root / 'experiments/monitoring/jobs'
            jobs.mkdir(parents=True)
            for date in ('20990101', '20990102'):
                eid = f'nav-robothor-embclip-light-gru1-formal5m-sjc-v1-seed0-{date}'
                (jobs / (eid + '.json')).write_text(json.dumps(dict(
                    experiment_id=eid, host='sjc-remote', model=dict(family='gru1'),
                    execution=dict(seed=0))))
            with self.assertRaisesRegex(RuntimeError, 'Ambiguous'):
                campaign_runs(root, 'formal5m')


    def test_eleven_run_comparison_executes_without_stale_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'data/embclip-robothor/robothor-objectnav/val/episodes'
            data.mkdir(parents=True)
            ids = [f'episode-{i:04d}' for i in range(1800)]
            for index in range(15):
                with gzip.open(data / f'FloorPlan{index}.json.gz', 'wt') as stream:
                    json.dump([dict(id=e) for e in ids[index*120:(index+1)*120]], stream)
            common = ids[:1783]
            sel = ids[:1516]
            campaigns = [dict(experiment_id=f'{model}-{seed}', model=dict(family=model), seed=seed)
                         for model in ('gru1', 'gawf1') for seed in range(5)]
            reference = root / 'results/nav-robothor-embclip-val-expert-reference-sjc-v2-20261024'
            reference.mkdir(parents=True)
            (reference / 'summary.json').write_text(json.dumps(dict(status='complete', episodes_sha256='hash', reference_steps={e: 10 for e in sel})))
            (reference / 'exit.json').write_text('{"return_code": 0}')
            (reference / 'behaviour_metrics.json').write_text(json.dumps(dict(
                expert_summary_sha256='hash', common_episode_ids=common, sel_episode_ids=sel,
                sel_definition='fixture', runs=[dict(experiment_id=r['experiment_id'], metrics_sha256='hash', scores=dict(sel=1.0), episode_scores=[]) for r in campaigns])))
            def load(eid, family, seed, steps, **kwargs):
                return dict(experiment_id=eid, model=family, metrics_sha256='hash', tasks={e: (1.0, 0.5, 10) for e in common})
            def digest(path):
                return ('60cfea09121cde1b49f52a8ee644f49a07550b4eadbe1b828dcd10f0e8ddf4b1'
                        if path.name == 'behaviour_metrics.json' else 'hash')
            stream = io.StringIO()
            with patch.object(metrics, 'ROOT', root), patch.object(metrics, 'campaign_runs', return_value=campaigns), patch.object(metrics, 'load_run', side_effect=load) as loader, patch.object(metrics, 'sha', side_effect=digest), contextlib.redirect_stdout(stream):
                metrics.compare()
            output = json.loads(stream.getvalue())
            self.assertEqual(loader.call_count, 11)
            self.assertEqual(len(output['runs']), 11)
            self.assertEqual(output['aggregates']['gru1']['n'], 5)
            self.assertEqual(output['aggregates']['gawf1']['n'], 5)
            self.assertEqual(output['aggregates']['official']['n'], 1)

class EntryPointChecks(unittest.TestCase):
    def test_training_monitor_uses_all_identities_before_phase_filter(self):
        from scripts.training import monitor
        runs = [dict(experiment_id='train', mode='formal_train'),
                dict(experiment_id='evaluation', mode='checkpoint_eval')]
        stream = io.StringIO()
        with patch.object(monitor, 'manifests', return_value=runs), patch.object(monitor, 'tmux_states', return_value=({}, [])) as scheduler, patch.object(monitor, 'row', return_value=dict(status='running', experiment_id='train')), patch('sys.argv', ['monitor', '--host', 'sjc-remote', '--json']), contextlib.redirect_stdout(stream):
            monitor.main()
        scheduler.assert_called_once_with(runs)
        self.assertEqual([r['experiment_id'] for r in json.loads(stream.getvalue())['jobs']], ['train'])

    def test_analysis_monitor_reports_case_progress(self):
        from scripts.analysis import monitor
        with tempfile.TemporaryDirectory() as folder, patch.object(monitor, 'ROOT', Path(folder)):
            result = Path(folder) / 'results/test'
            result.mkdir(parents=True)
            (result / 'progress.json').write_text(json.dumps(dict(completed_cases=3, expected_cases=12)))
            run = dict(experiment_id='analysis', mode='gradcam', status='running', model='gawf1', units=[dict(result_path='results/test')])
            row = monitor.row(run, {}, {})
            self.assertEqual((row['global_frames'], row['target_global_frames'], row['unit']), (3, 12, 'cases'))

    def test_media_and_gradcam_import_as_frozen_workers(self):
        import importlib.util
        from scripts.analysis.embclip import media, gradcam
        for source in (media, gradcam):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                leaf = root / 'results/unit'
                leaf.mkdir(parents=True)
                path = leaf / 'runner.py'
                path.write_text(Path(source.__file__).read_text())
                spec = importlib.util.spec_from_file_location('frozen_worker', path)
                worker = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(worker)
                self.assertEqual(worker.ROOT, root.resolve())


if __name__ == "__main__":
    unittest.main()
