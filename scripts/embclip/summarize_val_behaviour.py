"""Shared-episode planar curvature and expert-normalized SEL for light models.

Curvature is the time mean of the magnitude of the planar formula used by
Codebook, using first/second numpy finite differences on logged x-z positions.
Samples with planar speed <=1e-4 m/step contribute zero to avoid 0/0. Pure
rotations are not geometric curvature; report rotation counts separately.
SEL uses the same successful noise-free simulator-expert reference for all runs.
It is not a guarantee of a globally shortest action count.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def planar_curvature(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite planar coordinates')
    if len(points) < 3:
        return 0.0
    first = np.gradient(points, axis=0)
    second = np.gradient(first, axis=0)
    speed = np.linalg.norm(first, axis=1)
    value = np.zeros(len(points), dtype=np.float64)
    moving = speed > 1e-4
    value[moving] = np.abs(first[:, 0]*second[:, 1] - first[:, 1]*second[:, 0])[moving] / speed[moving]**3
    return float(value.mean())


def self_test():
    x = np.linspace(0, 5, 100)
    straight = np.column_stack((x, 2*x))
    assert planar_curvature(straight) < 1e-10
    assert planar_curvature(np.zeros((20, 2))) == 0
    assert planar_curvature(np.zeros((2, 2))) == 0
    angle = np.linspace(0, 2*np.pi, 2001)
    circle = 2*np.column_stack((np.cos(angle), np.sin(angle)))
    assert abs(planar_curvature(circle)-.5) < .001
    rotation = np.array([[0., -1.], [1., 0.]])
    assert abs(planar_curvature(circle)-planar_curvature(circle @ rotation + [10., 7.])) < 1e-9
    assert .5 * 10 / max(10, 20) == .25
    print('PASS straight, stationary, short, circle, rigid transform and SEL arithmetic')


def collect(root, reference):
    ref = json.loads((reference / 'summary.json').read_text())
    if ref['status'] != 'complete' or json.loads((reference / 'exit.json').read_text())['return_code'] != 0:
        raise RuntimeError('Expert reference incomplete')
    if sha(reference / 'episodes.jsonl') != ref['episodes_sha256']:
        raise RuntimeError('Expert trajectory hash mismatch')
    refs = ref['reference_steps']
    if any(not isinstance(n, int) or n < 1 for n in refs.values()):
        raise RuntimeError('Invalid expert reference lengths')
    expected = {e['id'] for p in (root / 'data/embclip-robothor/robothor-objectnav/val/episodes').glob('FloorPlan*.json.gz')
                for e in json.load(gzip.open(p, 'rt'))}
    if len(expected) != 1800:
        raise RuntimeError('Unexpected Val split')
    runs = []
    for model in ('gru1', 'gawf1'):
        for seed in range(5):
            date = '20261022' if seed < 2 else '20261023'
            rid = f'nav-robothor-embclip-light-{model}-eval-val1800-sjc-v1-seed{seed}-{date}'
            mp = root / 'experiments/monitoring/jobs' / (rid+'.json')
            manifest = json.loads(mp.read_text())
            out = root / manifest['units'][0]['result_path']
            summary = json.loads((out / 'summary.json').read_text())
            if (manifest['status'] not in ('complete', 'partial')
                    or manifest['completion_evidence']['return_code'] != 0
                    or json.loads((out / 'exit.json').read_text())['return_code'] != 0
                    or manifest['seed'] != seed or manifest['model']['family'] != model
                    or summary['checkpoint_steps'] != 5001216
                    or summary['checkpoint_sha256'] != manifest['source_checkpoint']['sha256']
                    or summary['ai2thor_version'] != '5.0.0'):
                raise RuntimeError('Run identity/exit mismatch: '+rid)
            metric_path = root / summary['metrics_path']
            if sha(metric_path) != summary['metrics_sha256']:
                raise RuntimeError('Metric hash mismatch: '+rid)
            metrics = json.loads(metric_path.read_text())
            if len(metrics) != 1 or int(metrics[0]['training_steps']) != 5001216:
                raise RuntimeError('Unexpected metric checkpoint')
            tasks = {t['task_info']['id']: t for t in metrics[0]['tasks']}
            if len(tasks) != len(metrics[0]['tasks']) or len(tasks) != summary['episode_count'] or not tasks.keys() <= expected:
                raise RuntimeError('Episode coverage mismatch')
            runs.append(dict(model=model, seed=seed, experiment_id=rid, tasks=tasks,
                checkpoint_sha256=summary['checkpoint_sha256'], metrics_sha256=summary['metrics_sha256']))
    common = set.intersection(*(set(r['tasks']) for r in runs))
    selected = set(json.loads((reference / 'episode_ids.json').read_text()))
    if selected != common or len(common) != 1783:
        raise RuntimeError('Expert selection does not match all ten model runs')
    sel_ids = common & refs.keys()
    if not sel_ids:
        raise RuntimeError('No usable expert references')
    for run in runs:
        values = []
        for episode_id in sorted(common):
            task = run['tasks'][episode_id]
            info = task['task_info']
            actions, path = info['taken_actions'], info['followed_path']
            if len(actions) != task['ep_length'] or len(path) != 1 + sum(a != 'End' for a in actions):
                raise RuntimeError('Action/pose alignment mismatch: '+episode_id)
            points = [[p['x'], p['z']] for p in path]
            steps = int(task['ep_length'])
            success = float(task['success'])
            reference_steps = refs.get(episode_id)
            values.append(dict(episode_id=episode_id, success=success, spl=float(task['spl']),
                episode_length=steps, curvature=planar_curvature(points),
                rotation_count=sum(a in ('RotateLeft','RotateRight') for a in actions),
                expert_reference_steps=reference_steps,
                sel=success*reference_steps/max(reference_steps,steps) if reference_steps else None))
        run.pop('tasks')
        run['scores'] = {metric: statistics.mean(r[metric] for r in values) for metric in
            ('success', 'spl', 'episode_length', 'curvature', 'rotation_count')}
        run['scores']['sel'] = statistics.mean(r['sel'] for r in values if r['sel'] is not None)
        run['episode_scores'] = values
    stats = {}
    for model in ('gru1', 'gawf1'):
        selected_runs = [r for r in runs if r['model'] == model]
        stats[model] = {}
        for metric in selected_runs[0]['scores']:
            factor = 100 if metric in ('success','spl','sel') else 1
            values = [r['scores'][metric]*factor for r in selected_runs]
            stats[model][metric] = dict(mean=statistics.mean(values), std=statistics.stdev(values), per_seed=values)
    return dict(common_episode_count=len(common), common_episode_ids=sorted(common),
        common_episode_ids_sha256=hashlib.sha256('\n'.join(sorted(common)).encode()).hexdigest(),
        sel_episode_count=len(sel_ids), sel_episode_ids=sorted(sel_ids),
        missing_expert_episode_ids=sorted(common-sel_ids), expert_summary_sha256=sha(reference / 'summary.json'),
        curvature_definition='Mean absolute planar curvature, numpy.gradient twice on logged x-z positions at action times; speed<=1e-4 m/step contributes 0; no interpolation or smoothing; 1/m.',
        stationary_limit='In-place rotations have zero planar displacement and cannot be assessed by position curvature. Rotation counts and episode lengths are retained separately.',
        sel_definition='Mean success * w/max(w,e); w is the shared noise-free ObjectNavExpertAction reference count, e is actual policy actions; both include End. Not guaranteed globally shortest or Codebook-equivalent.',
        aggregation='Equal-weight mean and sample std (ddof=1) across five training seeds. SR, SPL and SEL are percentages.',
        statistics=stats, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.root is None or args.reference is None:
            parser.error('--root and --reference required')
        print(json.dumps(collect(args.root.resolve(), args.reference.resolve()), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
