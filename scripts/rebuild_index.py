"""Explicit registry maintenance; task lookups continue to read one exact manifest."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'experiments/monitoring'
rows, active = [], []
for path in sorted((root / 'jobs').glob('*.json')):
    if path.name.startswith('._'):
        continue  # macOS tar metadata is not a manifest.
    run = json.loads(path.read_text())
    status = run['status']
    evidence = run.get('completion_evidence') or {}
    if run.get('mode') == 'train-smoke':
        progress = '%s/%s updates' % (evidence.get('completed_updates', 0), run['budget']['updates'])
    elif run.get('mode') == 'train_smoke':
        progress = '%s/%s stages' % (evidence.get('completed_stages', 0), run['expected_units'])
    elif run.get('mode') == 'checkpoint_eval_smoke':
        progress = '%s/%s episodes' % (evidence.get('episode_count', 0), run['expected_units'])
    elif run.get('mode') == 'preflight':
        progress = '%s/1 checks' % int(status == 'complete')
    elif run.get('mode') == 'formal_train':
        target = run['execution'].get('steps_budget') or run['execution'].get('target_steps') or run['expected_units']
        progress = '%s/%s steps' % (evidence.get('global_frames') or 0, target)
    else:
        done = sum(unit.get('metrics', {}).get('completed_episodes', 0) for unit in run.get('units', []))
        progress = '%s/%s episodes' % (done, run.get('expected_episodes', '?'))
    rows.append('| %s | %s | %s | %s |' % (run['experiment_id'], status, run['host'], progress))
    if status in ('submitted', 'running', 'recovering', 'paused', 'prepared'):
        active.append({'experiment_id': run['experiment_id'], 'host': run['host'], 'status': status,
                       'job_id': run.get('execution', {}).get('job_id')})
(root / 'JOBS.md').write_text('# Navigation experiment history\n\n| Experiment ID | Status | Host | Progress |\n| --- | --- | --- | --- |\n' + '\n'.join(rows) + '\n')
(root / 'active_jobs.json').write_text(json.dumps(active, indent=2) + '\n')
print('Indexed %d experiments; %d active' % (len(rows), len(active)))
