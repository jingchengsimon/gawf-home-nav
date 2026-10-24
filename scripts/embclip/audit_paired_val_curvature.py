"""Audit existing curvature and compare paired successful Val trajectories.

Read-only postprocessing. Primary success cohorts are defined before comparing
scores: per-seed GRU/GaWF joint success, and success in all ten runs. A sensitivity
analysis removes consecutive stationary positions and excludes episodes lacking
three moving positions in either paired model. No policy is rerun or altered.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_positions(points, minimum_distance=1e-4):
    points = np.asarray(points, dtype=float)
    keep = [0]
    for index in range(1, len(points)):
        if np.linalg.norm(points[index] - points[keep[-1]]) > minimum_distance:
            keep.append(index)
    return points[keep]


def bending_density(points):
    """Polyline turning radians per traveled meter, without idle padding.

    Ignore consecutive displacements <=1cm. Require two segments and at least
    0.5m of movement. This diagnostic is distinct from differential curvature.
    """
    points = compact_positions(points, minimum_distance=.01)
    if len(points) < 3:
        return None
    segments = np.diff(points, axis=0)
    length = np.linalg.norm(segments, axis=1).sum()
    if length < .5:
        return None
    first, second = segments[:-1], segments[1:]
    angle = np.arctan2(np.abs(first[:,0]*second[:,1]-first[:,1]*second[:,0]),
                       (first*second).sum(axis=1))
    return float(angle.sum()/length)


def aggregate(rows):
    if not rows:
        return None
    metrics = ['curvature', 'movement_curvature', 'rotation_count', 'episode_length',
               'spl', 'sel', 'travelled_m', 'yaw_turn_degrees', 'stationary_fraction',
               'near_stationary_episode', 'failed_end_within10', 'position_count', 'bending_density']
    return {key: dict(mean=statistics.mean(r[key] for r in rows if r[key] is not None),
                     count=sum(r[key] is not None for r in rows))
            for key in metrics if any(r[key] is not None for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-behaviour-sha256', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'results/nav-robothor-embclip-val-expert-reference-sjc-v2-20261024/behaviour_metrics.json'
    if sha(source) != args.expected_behaviour_sha256:
        raise RuntimeError('Verified behavior snapshot changed')
    previous = json.loads(source.read_text())
    implementation = source.parent / 'analysis.py'
    manifest = json.loads((root / 'experiments/monitoring/jobs/nav-robothor-embclip-val-expert-reference-sjc-v2-20261024.json').read_text())
    if sha(implementation) != manifest['analysis_sha256']:
        raise RuntimeError('Original curvature implementation changed')
    spec = importlib.util.spec_from_file_location('original_behaviour', implementation)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    curvature = module.planar_curvature
    assert curvature(np.zeros((40, 2))) == 0.0
    # A curved path followed by idle samples illustrates action-time averaging.
    theta = np.linspace(0, np.pi, 60)
    arc = np.column_stack((np.cos(theta), np.sin(theta)))
    idle = np.vstack([arc, np.repeat(arc[-1:], 180, axis=0)])
    assert curvature(idle) < curvature(arc)
    assert np.array_equal(compact_positions(idle), arc)
    assert abs(bending_density(arc)-bending_density(idle)) < 1e-12
    assert abs(bending_density([[0.,0.],[1.,0.],[0.,0.]])-np.pi/2) < 1e-12
    runs = {}
    for run in previous['runs']:
        model, seed = run['model'], run['seed']
        rid = run['experiment_id']
        mp = root / 'experiments/monitoring/jobs' / (rid+'.json')
        m = json.loads(mp.read_text())
        summary = json.loads((root / m['units'][0]['result_path'] / 'summary.json').read_text())
        path = root / summary['metrics_path']
        if (m['status'] not in ('complete','partial') or m['completion_evidence']['return_code'] != 0
                or m['seed'] != seed or m['model']['family'] != model
                or summary['checkpoint_sha256'] != run['checkpoint_sha256']
                or sha(path) != run['metrics_sha256']):
            raise RuntimeError('Run identity/hash mismatch: '+rid)
        tasks = json.loads(path.read_text())[0]['tasks']
        tasks = {t['task_info']['id']: t for t in tasks}
        scores = {}
        for old in run['episode_scores']:
            eid = old['episode_id']
            task = tasks[eid]
            info = task['task_info']
            positions = info['followed_path']
            points = np.array([[p['x'],p['z']] for p in positions])
            delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
            derivative_speed = np.linalg.norm(np.gradient(points, axis=0), axis=1) if len(points)>1 else np.array([0.])
            movement = compact_positions(points)
            old_curvature = curvature(points)
            if abs(old_curvature-old['curvature']) > 1e-12:
                raise RuntimeError('Original curvature not reproduced')
            yaws = np.array([p['rotation']['y'] for p in positions])
            yaw_turn = np.abs((np.diff(yaws)+180)%360-180).sum()
            row = dict(old, movement_curvature=curvature(movement) if len(movement)>=3 else None,
                bending_density=bending_density(points),
                travelled_m=float(delta.sum()), yaw_turn_degrees=float(yaw_turn),
                stationary_fraction=float(np.mean(derivative_speed<=1e-4)),
                near_stationary_episode=int(delta.sum()<.01),
                failed_end_within10=int(not old['success'] and task['ep_length']<=10 and info['taken_actions'][-1]=='End'),
                position_count=len(points), moving_position_count=len(movement))
            scores[eid] = row
        runs[model,seed] = scores
    if set(runs) != {(m,s) for m in ('gru1','gawf1') for s in range(5)}:
        raise RuntimeError('Expected ten runs')
    common = set(previous['common_episode_ids'])
    joint = {seed: {eid for eid in common if runs['gru1',seed][eid]['success'] and runs['gawf1',seed][eid]['success']}
             for seed in range(5)}
    strict = set.intersection(*joint.values())
    motion = {seed: {eid for eid in joint[seed] if all(runs[m,seed][eid]['movement_curvature'] is not None for m in ('gru1','gawf1'))}
              for seed in range(5)}
    meaningful = {seed: {eid for eid in joint[seed] if all(runs[m,seed][eid]['bending_density'] is not None for m in ('gru1','gawf1'))}
                  for seed in range(5)}
    cohorts = {'all_val': {s: common for s in range(5)}, 'paired_success': joint,
               'all_ten_success': {s: strict for s in range(5)}, 'paired_success_both_move': motion,
               'paired_success_meaningful_motion': meaningful}
    report = {}
    for name, selections in cohorts.items():
        rows = []
        for seed, ids in selections.items():
            for model in ('gru1','gawf1'):
                selected = [runs[model,seed][i] for i in sorted(ids)]
                rows.append(dict(model=model,seed=seed,episodes=len(selected),scores=aggregate(selected)))
        stats = {}
        for model in ('gru1','gawf1'):
            selected = [r for r in rows if r['model']==model and r['scores']]
            stats[model] = {}
            for key in selected[0]['scores'] if selected else []:
                values = [r['scores'][key]['mean'] for r in selected if key in r['scores']]
                factor = 100 if key in ('spl','sel') else 1
                stats[model][key] = dict(mean=statistics.mean(values)*factor,
                    sample_std=statistics.stdev(values)*factor if len(values)>1 else None,
                    per_seed=[v*factor for v in values])
        report[name] = dict(episode_ids_by_seed={str(s):sorted(ids) for s,ids in selections.items()},
            counts_by_seed={str(s):len(ids) for s,ids in selections.items()}, runs=rows, statistics=stats)
    examples = {}
    for model in ('gru1','gawf1'):
        candidates = [dict(seed=s,**r) for s in range(5) for r in runs[model,s].values()
                      if r['travelled_m']<1e-4 and r['rotation_count']>=20]
        examples[model] = sorted(candidates,key=lambda r:r['rotation_count'],reverse=True)[:2]
    print(json.dumps(dict(status='pass',source_behaviour_sha256=sha(source),original_curvature_source_sha256=sha(implementation),
        original_curvature_reproduced_episodes=17830,
        thresholds=dict(stationary_speed_m_per_action=1e-4,near_stationary_episode_m=.01,early_failed_end_max_actions=10),
        definitions=dict(primary='Original action-time planar curvature reproduced exactly; stationary-speed samples contribute zero.',
            movement_sensitivity='Drop consecutive positions within 1e-4 m; same curvature formula; paths with fewer than three moving positions are undefined and excluded symmetrically.',
            bending_density='Separate robust polyline diagnostic: total absolute positional turning angle / traveled meters, ignoring <=1cm displacements, requiring >=0.5m and two segments in both paired models. Does not measure in-place yaw.',
            paired_success='Same training seed index, same episode ID, both successful. Cohort can differ across seed indices.',
            all_ten_success='Identical episode IDs on which all ten model runs succeeded.',
            uncertainty='Equal-weight five seed means; sample SD (ddof=1). These success-conditioned subsets are selection-biased and not benchmark success estimates.'),
        cohorts=report,zero_translation_rotation_examples=examples,
        idle_padding_test=dict(curvature_arc=curvature(arc),curvature_arc_plus180_idle_positions=curvature(idle))),indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
