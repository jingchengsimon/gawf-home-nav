"""Run bounded expert-reference shards and merge their verified receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    mp = root / args.manifest
    m = json.loads(mp.read_text())
    out = root / m['result_path']
    if m['status'] != 'prepared' or sha(__file__) != m['runner_sha256'] or out != Path(__file__).resolve().parent:
        raise RuntimeError('Unexpected batch identity')
    prior = root / m['prior_episodes_path']
    if sha(prior) != m['prior_episodes_sha256']:
        raise RuntimeError('Prior reference records changed')
    rows = [json.loads(line) for line in prior.read_text().splitlines()]
    expected = set(json.loads((out / 'episode_ids.json').read_text()))
    if len(rows) != len({r['episode_id'] for r in rows}):
        raise RuntimeError('Duplicate prior episode')
    prior_mp = root / m['prior_manifest']
    old = json.loads(prior_mp.read_text())
    if old['status'] != 'failed' or 'KeyboardInterrupt' not in old.get('failure', ''):
        raise RuntimeError('Unexpected prior stop state')
    old.update(status='superseded', superseded_by=m['experiment_id'],
        retained_partial_episode_count=len(rows), reason='Bounded serial run migrated to eight shards; partial records reused unchanged.')
    old['units'][0]['status'] = 'superseded'
    save(prior_mp, old)
    m['status'] = 'running'
    save(mp, m)
    started = time.monotonic()

    def run(unit):
        child_mp = root / unit['manifest_path']
        child = json.loads(child_mp.read_text())
        leaf = root / child['units'][0]['result_path']
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with (leaf / 'run.log').open('w') as log:
            rc = subprocess.call([str(root / '.envs/embclip-robothor/bin/python'), '-B', '-u', str(leaf / 'runner.py'),
                '--root', str(root), '--manifest', str(child_mp)], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            raise RuntimeError('Expert shard failed: '+child['experiment_id'])
        summary = json.loads((leaf / 'summary.json').read_text())
        receipt = json.loads(child_mp.read_text())
        if summary['status'] != 'complete' or sha(leaf / 'summary.json') != receipt['completion_evidence']['summary_sha256']:
            raise RuntimeError('Shard summary mismatch')
        if sha(leaf / 'episodes.jsonl') != summary['episodes_sha256']:
            raise RuntimeError('Shard episode hash mismatch')
        return unit, [json.loads(line) for line in (leaf / 'episodes.jsonl').read_text().splitlines()]

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(run, u) for u in m['units']]
            for future in as_completed(futures):
                unit, completed = future.result()
                rows.extend(completed)
                unit['status'] = 'complete'
                save(mp, m)
                save(out / 'progress.json', dict(status='running', completed_episodes=len(rows),
                    expected_episodes=len(expected), valid_references=sum(r['success'] for r in rows)))
                print('REFERENCE_SHARD_COMPLETE', unit['name'], len(rows), '/', len(expected), flush=True)
        ids = [r['episode_id'] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise RuntimeError('Batch episode coverage mismatch')
        rows.sort(key=lambda r: r['episode_id'])
        (out / 'episodes.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows))
        refs = {r['episode_id']: r['reference_actions'] for r in rows if r['success']}
        summary = dict(status='complete', expected_episodes=len(expected), checked_episodes=len(rows),
            valid_reference_count=len(refs), reference_steps=refs,
            failures=[dict(episode_id=r['episode_id'], failure=r['failure']) for r in rows if not r['success']],
            elapsed_seconds=time.monotonic()-started, episodes_sha256=sha(out / 'episodes.jsonl'),
            reference='AI2-THOR ObjectNavExpertAction, noiseless reference dynamics, 1m visibility; End included.',
            globally_shortest_guaranteed=False, codebook_exact_reproduction=False, policy_inputs_modified=False)
        save(out / 'summary.json', summary)
        save(out / 'exit.json', dict(return_code=0))
        m['status'] = 'complete'
        m['completion_evidence'] = dict(return_code=0, checked_episodes=len(rows), valid_reference_count=len(refs),
            summary_sha256=sha(out / 'summary.json'))
        save(mp, m)
        print('REFERENCE_BATCH_COMPLETE', len(refs), '/', len(rows), flush=True)
    except BaseException as error:
        m['status'] = 'failed'
        m['failure'] = repr(error)
        save(mp, m)
        save(out / 'exit.json', dict(return_code=1, error=repr(error)))
        raise
    finally:
        subprocess.run([sys.executable, '-B', 'scripts/rebuild_index.py'], cwd=root, check=True)


if __name__ == '__main__':
    main()
