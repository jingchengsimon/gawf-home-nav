"""Collect and compare current RoboTHOR training, validation and behavior metrics."""
from __future__ import annotations
import argparse
from datetime import datetime
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
MODELS = ("gru1", "gawf1")
SEEDS = tuple(range(5))
SEEDS_BY_MODEL = {model: SEEDS for model in MODELS}

def campaign_runs(root: Path, phase: str) -> list[dict]:
    """Return one manifest per model/seed, rejecting missing or duplicate identities."""
    if phase not in {"formal5m", "eval-val1800"}:
        raise ValueError("Expected formal5m or eval-val1800")
    selected = {}
    folder = root / "experiments/monitoring/jobs"
    for path in sorted(folder.glob(f"nav-robothor-embclip-light-*-{phase}-sjc-*.json")):
        run = json.loads(path.read_text())
        model = run["model"]["family"]
        seed = run.get("seed", run.get("execution", {}).get("seed"))
        if model not in MODELS or seed not in SEEDS:
            continue
        identity = (model, seed)
        if (run["experiment_id"] != path.stem or run["host"] != "sjc-remote"
                or identity in selected):
            raise RuntimeError(f"Ambiguous campaign identity: {path}")
        selected[identity] = run
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    if set(selected) != expected:
        raise RuntimeError(f"Missing campaign runs: {sorted(expected - selected.keys())}")
    return [selected[(model, seed)] for model in MODELS for seed in SEEDS]



def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def load_run(experiment_id, family, seed, steps, raw_tasks=False, root=None):
    root = ROOT if root is None else Path(root)
    manifest_path = root / 'experiments/monitoring/jobs' / (experiment_id + '.json')
    manifest = json.loads(manifest_path.read_text())
    out = root / manifest['units'][0]['result_path']
    summary_path = out / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if ((family != 'official' and (manifest['model']['family'] != family or manifest['seed'] != seed))
            or manifest['status'] not in ('partial', 'complete')
            or summary['status'] != manifest['status']
            or manifest['completion_evidence']['return_code'] != 0
            or json.loads((out / 'exit.json').read_text())['return_code'] != 0
            or summary['checkpoint_steps'] != steps
            or summary['checkpoint_sha256'] != manifest['source_checkpoint']['sha256']
            or summary['ai2thor_version'] != '5.0.0'
            or summary['ai2thor_commit'] != 'f0825767cd50d69f666c7f282e54abfe58f1e917'):
        raise RuntimeError('Incomplete or incompatible evaluation: ' + experiment_id)
    metrics_path = root / summary['metrics_path']
    if sha(metrics_path) != summary['metrics_sha256']:
        raise RuntimeError('Metrics hash mismatch')
    rows = json.loads(metrics_path.read_text())
    if len(rows) != 1 or int(rows[0]['training_steps']) != steps:
        raise RuntimeError('Wrong checkpoint in metrics')
    tasks = rows[0]['tasks']
    mapping = {t['task_info']['id']: (float(t['success']), float(t['spl']), int(t['ep_length'])) for t in tasks}
    if len(tasks) != len(mapping) or len(tasks) != summary['episode_count']:
        raise RuntimeError('Duplicate or missing task records')
    expected_summary_hash = manifest['completion_evidence'].get('summary_sha256')
    if expected_summary_hash and sha(summary_path) != expected_summary_hash:
        raise RuntimeError('Summary hash mismatch')
    if any(t['ep_length'] != len(t['task_info']['taken_actions']) or t['ep_length'] < 1 for t in tasks):
        raise RuntimeError('Action count mismatch')
    for index, metric in enumerate(('success_rate', 'spl')):
        if not math.isclose(statistics.mean(v[index] for v in mapping.values()), summary[metric], abs_tol=1e-12):
            raise RuntimeError('Per-episode and summary scores disagree')
    return dict(experiment_id=experiment_id, model=family, training_seed=seed,
        training_steps=steps, episode_count=len(tasks), summary_sha256=sha(summary_path),
        metrics_sha256=sha(metrics_path), checkpoint_sha256=summary['checkpoint_sha256'], tasks=({t['task_info']['id']: t for t in tasks} if raw_tasks else mapping))



def compare():
    files = sorted((ROOT / 'data/embclip-robothor/robothor-objectnav/val/episodes').glob('FloorPlan*.json.gz'))
    expected = {e['id'] for p in files for e in json.load(gzip.open(p, 'rt'))}
    if len(files) != 15 or len(expected) != 1800:
        raise RuntimeError('Unexpected validation data')
    runs = []
    for manifest in campaign_runs(ROOT, "eval-val1800"):
        runs.append(load_run(manifest["experiment_id"], manifest["model"]["family"],
                             manifest["seed"], 5_001_216))
    runs.append(load_run('nav-robothor-embclip-official130m-eval-val1800-sjc-v1-20261024', 'official', None, 130091717))
    for run in runs:
        if not set(run['tasks']) <= expected:
            raise RuntimeError('Unknown episode ID')
    common = set.intersection(*(set(r['tasks']) for r in runs))
    if not common:
        raise RuntimeError('No common episode IDs')
    reference = ROOT / 'results/nav-robothor-embclip-val-expert-reference-sjc-v2-20261024'
    behaviour_path = reference / 'behaviour_metrics.json'
    if sha(behaviour_path) != '60cfea09121cde1b49f52a8ee644f49a07550b4eadbe1b828dcd10f0e8ddf4b1':
        raise RuntimeError('Previously verified behavior snapshot changed')
    behaviour = json.loads(behaviour_path.read_text())
    expert = json.loads((reference / 'summary.json').read_text())
    if (expert['status'] != 'complete'
            or json.loads((reference / 'exit.json').read_text())['return_code'] != 0
            or sha(reference / 'summary.json') != behaviour['expert_summary_sha256']
            or sha(reference / 'episodes.jsonl') != expert['episodes_sha256']
            or common != set(behaviour['common_episode_ids'])):
        raise RuntimeError('Expert reference identity/coverage mismatch')
    refs = expert['reference_steps']
    if any(not isinstance(n, int) or n < 1 for n in refs.values()):
        raise RuntimeError('Invalid expert reference length')
    sel_ids = common & refs.keys()
    if sel_ids != set(behaviour['sel_episode_ids']) or len(sel_ids) != 1516:
        raise RuntimeError('SEL comparison set changed')
    old_runs = {r['experiment_id']: r for r in behaviour['runs']}
    for run in runs:
        tasks = run.pop('tasks')
        run['common_sr'] = statistics.mean(tasks[e][0] for e in common)
        run['common_spl'] = statistics.mean(tasks[e][1] for e in common)
        scores = {e: tasks[e][0]*refs[e]/max(refs[e], tasks[e][2]) for e in sel_ids}
        run['common_sel'] = statistics.mean(scores.values())
        if run['model'] != 'official':
            old = old_runs[run['experiment_id']]
            if (old['metrics_sha256'] != run['metrics_sha256']
                    or not math.isclose(old['scores']['sel'], run['common_sel'], abs_tol=1e-12)):
                raise RuntimeError('Previously verified lightweight SEL not reproduced')
            for row in old['episode_scores']:
                if row['episode_id'] in scores and not math.isclose(row['sel'], scores[row['episode_id']], abs_tol=1e-12):
                    raise RuntimeError('Per-episode lightweight SEL not reproduced')
        run['missing_episode_ids'] = sorted(expected-set(tasks))
    aggregate = {}
    for model in ('official', 'gru1', 'gawf1'):
        selected = [r for r in runs if r['model'] == model]
        aggregate[model] = dict(n=len(selected))
        for metric in ('common_sr', 'common_spl', 'common_sel'):
            values = [r[metric] * 100 for r in selected]
            aggregate[model][metric] = dict(mean_percent=statistics.mean(values),
                sample_std_percentage_points=statistics.stdev(values) if len(values)>1 else None,
                seed_scores_percent=values)
    print(json.dumps(dict(expected_episode_count=1800, common_episode_count=len(common),
        common_episode_ids=sorted(common), common_episode_ids_sha256=hashlib.sha256('\n'.join(sorted(common)).encode()).hexdigest(),
        sel_episode_count=len(sel_ids), sel_episode_ids=sorted(sel_ids),
        sel_episode_ids_sha256=hashlib.sha256('\n'.join(sorted(sel_ids)).encode()).hexdigest(),
        expert_summary_sha256=sha(reference / 'summary.json'),
        behaviour_source_sha256=sha(behaviour_path), sel_definition=behaviour['sel_definition'],
        standard_deviation='Across five training seeds; sample standard deviation (ddof=1). Official checkpoint has n=1; no seed std.',
        protocol=dict(ai2thor='5.0.0', build='f0825767cd50d69f666c7f282e54abfe58f1e917',
            max_steps=500, evaluation_seed=0, raw_rgb=[400,300], policy_rgb=[224,224], num_test_processes=15),
        runs=runs, aggregates=aggregate), indent=2, sort_keys=True))



def planar_curvature(points):
    import numpy as np

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
    import numpy as np

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



def collect_behaviour(root, reference):
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
    for manifest in campaign_runs(root, "eval-val1800"):
        model, seed = manifest["model"]["family"], manifest["seed"]
        rid = manifest["experiment_id"]
        record = load_run(rid, model, seed, 5_001_216, raw_tasks=True, root=root)
        runs.append(dict(model=model, seed=seed, experiment_id=rid, tasks=record["tasks"],
                         checkpoint_sha256=record["checkpoint_sha256"],
                         metrics_sha256=record["metrics_sha256"]))
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



def collect_validation():

    root = ROOT
    files = sorted((root / "data/embclip-robothor/robothor-objectnav/val/episodes").glob("*.json.gz"))
    expected = {e["id"] for path in files for e in json.load(gzip.open(path, "rt"))}
    if len(files) != 15 or len(expected) != 1800:
        raise RuntimeError("Expected 15 validation scenes and 1,800 unique episodes")
    rows = []
    for run in campaign_runs(root, "eval-val1800"):
        row = load_run(run["experiment_id"], run["model"]["family"], run["seed"], 5_001_216)
        summary = json.loads((root / run["units"][0]["result_path"] / "summary.json").read_text())
        if (not row["tasks"].keys() <= expected
                or sorted(expected - row["tasks"].keys()) != summary["missing_episode_ids"]):
            raise RuntimeError("Validation episode coverage mismatch")
        row.update(seed=row.pop("training_seed"), status=run["status"])
        rows.append(row)
    common = set.intersection(*(set(row["tasks"]) for row in rows))
    if not common:
        raise RuntimeError("No common validation episodes")
    for row in rows:
        tasks = row.pop("tasks")
        row["common_sr"] = statistics.mean(tasks[eid][0] for eid in common)
        row["common_spl"] = statistics.mean(tasks[eid][1] for eid in common)
    snapshot = dict(observed_at=datetime.now().astimezone().isoformat(),
                    expected_episode_count=1800, common_episode_count=len(common),
                    common_episode_ids_sha256=hashlib.sha256("\n".join(sorted(common)).encode()).hexdigest(),
                    selected_seeds={model: list(seeds) for model, seeds in SEEDS_BY_MODEL.items()},
                    runs=rows)
    return snapshot



def collect_training():
    rows = []

    for run in campaign_runs(ROOT, "formal5m"):
        model, seed = run["model"]["family"], run["execution"]["seed"]
        eid = run["experiment_id"]
        ids = [eid]
        prior = run.get("resume_from", {}).get("checkpoint_path")
        if prior:
            parts = Path(prior).parts
            index = parts.index("results")
            ids.insert(0, parts[index + 1])
        points, sources = {}, []
        for identity in ids:
            for p in sorted((ROOT / "results" / identity).glob("**/train.log")):
                content = p.read_text(errors="replace")
                found = 0
                for line in content.splitlines():
                    match = re.search(r"TRAIN:\s+(\d+) rollout steps", line)
                    if match is None:
                        continue
                    step = int(match.group(1))
                    values = {}
                    for metric in ("success", "spl"):
                        m = re.search(r"\b" + metric + r"\s+([0-9.eE+-]+)", line)
                        if m is None:
                            raise RuntimeError(f"Missing {metric} in {p}")
                        values[metric] = float(m.group(1))
                    points[step] = {"step": step, **values}
                    found += 1
                if found:
                    sources.append({"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        if not points or max(points) != 5001216:
            raise RuntimeError(f"Incomplete source curve {model} {seed}")
        rows.append({"model": model, "seed": seed, "experiment_ids": ids,
                     "sources": sources, "points": [points[s] for s in sorted(points)]})
    return {"observed_at": datetime.now().astimezone().isoformat(),
                      "scope": "training completed episode means from AllenAct TRAIN records; not validation",
                      "runs": rows}


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("compare", "validation", "training", "behaviour", "self-test"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    if args.operation == "compare":
        compare()
        return
    if args.operation == "self-test":
        self_test()
        return
    if args.operation == "behaviour":
        if args.reference is None:
            parser.error("behaviour requires --reference")
        result = collect_behaviour(ROOT, args.reference.resolve())
    else:
        result = collect_validation() if args.operation == "validation" else collect_training()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
