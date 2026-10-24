"""Read verified Val outputs and compare all eleven runs on identical episodes.

Run on SJC from the project root, with this source supplied on stdin. Only JSON
is printed; experiment outputs are never modified.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path.cwd()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(experiment_id, family, seed, steps):
    manifest_path = ROOT / 'experiments/monitoring/jobs' / (experiment_id + '.json')
    manifest = json.loads(manifest_path.read_text())
    out = ROOT / manifest['units'][0]['result_path']
    summary_path = out / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if (manifest['status'] not in ('partial', 'complete')
            or summary['status'] != manifest['status']
            or manifest['completion_evidence']['return_code'] != 0
            or json.loads((out / 'exit.json').read_text())['return_code'] != 0
            or summary['checkpoint_steps'] != steps
            or summary['checkpoint_sha256'] != manifest['source_checkpoint']['sha256']
            or summary['ai2thor_version'] != '5.0.0'
            or summary['ai2thor_commit'] != 'f0825767cd50d69f666c7f282e54abfe58f1e917'):
        raise RuntimeError('Incomplete or incompatible evaluation: ' + experiment_id)
    metrics_path = ROOT / summary['metrics_path']
    if sha(metrics_path) != summary['metrics_sha256']:
        raise RuntimeError('Metrics hash mismatch')
    rows = json.loads(metrics_path.read_text())
    if len(rows) != 1 or int(rows[0]['training_steps']) != steps:
        raise RuntimeError('Wrong checkpoint in metrics')
    tasks = rows[0]['tasks']
    mapping = {t['task_info']['id']: (float(t['success']), float(t['spl'])) for t in tasks}
    if len(tasks) != len(mapping) or len(tasks) != summary['episode_count']:
        raise RuntimeError('Duplicate or missing task records')
    for index, metric in enumerate(('success_rate', 'spl')):
        if not math.isclose(statistics.mean(v[index] for v in mapping.values()), summary[metric], abs_tol=1e-12):
            raise RuntimeError('Per-episode and summary scores disagree')
    return dict(experiment_id=experiment_id, model=family, training_seed=seed,
        training_steps=steps, episode_count=len(tasks), summary_sha256=sha(summary_path),
        metrics_sha256=sha(metrics_path), checkpoint_sha256=summary['checkpoint_sha256'], tasks=mapping)


def main():
    files = sorted((ROOT / 'data/embclip-robothor/robothor-objectnav/val/episodes').glob('FloorPlan*.json.gz'))
    expected = {e['id'] for p in files for e in json.load(gzip.open(p, 'rt'))}
    if len(files) != 15 or len(expected) != 1800:
        raise RuntimeError('Unexpected validation data')
    runs = []
    for model in ('gru1', 'gawf1'):
        for seed in range(5):
            date = '20261022' if seed < 2 else '20261023'
            eid = f'nav-robothor-embclip-light-{model}-eval-val1800-sjc-v1-seed{seed}-{date}'
            runs.append(load_run(eid, model, seed, 5001216))
    runs.append(load_run('nav-robothor-embclip-official130m-eval-val1800-sjc-v1-20261024', 'official', None, 130091717))
    for run in runs:
        if not set(run['tasks']) <= expected:
            raise RuntimeError('Unknown episode ID')
    common = set.intersection(*(set(r['tasks']) for r in runs))
    if not common:
        raise RuntimeError('No common episode IDs')
    for run in runs:
        tasks = run.pop('tasks')
        run['common_sr'] = statistics.mean(tasks[e][0] for e in common)
        run['common_spl'] = statistics.mean(tasks[e][1] for e in common)
        run['missing_episode_ids'] = sorted(expected-set(tasks))
    aggregate = {}
    for model in ('official', 'gru1', 'gawf1'):
        selected = [r for r in runs if r['model'] == model]
        aggregate[model] = dict(n=len(selected))
        for metric in ('common_sr', 'common_spl'):
            values = [r[metric] * 100 for r in selected]
            aggregate[model][metric] = dict(mean_percent=statistics.mean(values),
                sample_std_percentage_points=statistics.stdev(values) if len(values)>1 else None,
                seed_scores_percent=values)
    print(json.dumps(dict(expected_episode_count=1800, common_episode_count=len(common),
        common_episode_ids=sorted(common), common_episode_ids_sha256=hashlib.sha256('\n'.join(sorted(common)).encode()).hexdigest(),
        standard_deviation='Across five training seeds; sample standard deviation (ddof=1). Official checkpoint has n=1; no seed std.',
        protocol=dict(ai2thor='5.0.0', build='f0825767cd50d69f666c7f282e54abfe58f1e917',
            max_steps=500, evaluation_seed=0, raw_rgb=[400,300], policy_rgb=[224,224], num_test_processes=15),
        runs=runs, aggregates=aggregate), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
