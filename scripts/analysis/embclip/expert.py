"""Collect evaluator-only expert reference lengths for shared RoboTHOR Val.

Uses the simulator's ObjectNavExpertAction with action noise disabled. This is
an expert-normalized SEL reference, not a claim of a globally shortest action
sequence or an exact reproduction of Codebook's unreleased expert.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    mp = root / args.manifest
    manifest = json.loads(mp.read_text())
    out = root / manifest['units'][0]['result_path']
    if (manifest['status'] != 'prepared' or sha(__file__) != manifest['runner_sha256']
            or out != Path(__file__).resolve().parent or (out / 'summary.json').exists()):
        raise RuntimeError('Unexpected source, result leaf or experiment state')
    ids_path = out / 'episode_ids.json'
    if sha(ids_path) != manifest['episode_ids_sha256']:
        raise RuntimeError('Episode selection changed')
    ids = json.loads(ids_path.read_text())
    episodes = {}
    for filename, digest in manifest['validation']['files'].items():
        path = root / filename
        if sha(path) != digest:
            raise RuntimeError('Validation data changed')
        for episode in json.load(gzip.open(path, 'rt')):
            episodes[episode['id']] = episode
    if not set(ids) <= episodes.keys() or len(ids) != len(set(ids)):
        raise RuntimeError('Invalid episode selection')
    selected = sorted((episodes[i] for i in ids), key=lambda e: (e['scene'], e['id']))
    import ai2thor
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    if ai2thor.__version__ != '5.0.0':
        raise RuntimeError('Wrong evaluation runtime')
    Controller.base_dir = property(lambda self: str(root / 'assets/embclip-robothor/ai2thor'))
    controller = None
    manifest['status'] = manifest['units'][0]['status'] = 'running'
    save(mp, manifest)
    rows = []
    started = time.monotonic()
    try:
        controller = Controller(platform=CloudRendering,
            commit_id='f0825767cd50d69f666c7f282e54abfe58f1e917', gpu_device=0,
            scene=selected[0]['scene'], width=400, height=300, gridSize=.25,
            snapToGrid=False, continuousMode=True, stochastic=True,
            applyActionNoise=False, rotateStepDegrees=30, visibilityDistance=1.0,
            agentMode='locobot', fieldOfView=63.453048374758716, renderDepthImage=False)
        scene = None
        for episode in selected:
            if episode['scene'] != scene:
                controller.reset(episode['scene'])
                scene = episode['scene']
            event = controller.step(action='TeleportFull', position=episode['initial_position'],
                rotation=dict(x=0, y=episode['initial_orientation'], z=0),
                horizon=episode.get('initial_horizon', 0), renderImage=False)
            actions, poses, visited = [], [], {}
            success = False
            failure = None
            if not event.metadata['lastActionSuccess']:
                failure = 'initial_teleport_failed'
            else:
                agent = event.metadata['agent']
                if max(abs(agent['position'][k]-episode['initial_position'][k]) for k in ('x','y','z')) > 1e-3:
                    raise RuntimeError('Initial pose mismatch')
                for _ in range(500):
                    if any(o['objectType'] == episode['object_type'] and o['visible']
                           for o in event.metadata['objects']):
                        actions.append('End')
                        success = True
                        break
                    agent = event.metadata['agent']
                    pose = dict(position=agent['position'], rotation=agent['rotation'], horizon=agent['cameraHorizon'])
                    poses.append(pose)
                    key = tuple(round(agent['position'][k], 3) for k in ('x','z')) + (
                        round(agent['rotation']['y'], 1), round(agent['cameraHorizon'], 1))
                    visited[key] = visited.get(key, 0) + 1
                    if visited[key] > 4:
                        failure = 'expert_repeated_state'
                        break
                    query = controller.step(action='ObjectNavExpertAction', objectType=episode['object_type'], renderImage=False)
                    action = query.metadata.get('actionReturn')
                    if not query.metadata['lastActionSuccess'] or action not in ('MoveAhead','RotateLeft','RotateRight','LookUp','LookDown'):
                        failure = dict(reason='expert_action_unavailable', returned=action,
                            error=query.metadata.get('errorMessage'))
                        break
                    actions.append(action)
                    event = controller.step(action=action, renderImage=False)
                if not success and failure is None:
                    failure = 'expert_500_action_limit'
            row = dict(episode_id=episode['id'], success=success,
                reference_actions=len(actions) if success else None,
                failure=failure, actions=actions, poses=poses)
            rows.append(row)
            with (out / 'episodes.jsonl').open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True)+'\n')
            save(out / 'progress.json', dict(status='running', completed_episodes=len(rows),
                expected_episodes=len(selected), valid_references=sum(r['success'] for r in rows)))
            if len(rows) % 25 == 0 or len(selected) < 25:
                print('EXPERT_PROGRESS', len(rows), '/', len(selected), 'valid', sum(r['success'] for r in rows), flush=True)
        valid = {r['episode_id']: r['reference_actions'] for r in rows if r['success']}
        summary = dict(status='complete', expected_episodes=len(selected), checked_episodes=len(rows),
            valid_reference_count=len(valid), reference_steps=valid,
            failures=[dict(episode_id=r['episode_id'], failure=r['failure']) for r in rows if not r['success']],
            elapsed_seconds=time.monotonic()-started, episodes_sha256=sha(out / 'episodes.jsonl'),
            reference='AI2-THOR ObjectNavExpertAction, noiseless reference dynamics, same task geometry, 1m visibility; End included.',
            maximum_reference_actions=500, actors_loaded=0, policy_inputs_modified=False,
            globally_shortest_guaranteed=False, codebook_exact_reproduction=False)
        save(out / 'summary.json', summary)
        save(out / 'exit.json', dict(return_code=0))
        manifest['status'] = manifest['units'][0]['status'] = 'complete'
        manifest['completion_evidence'] = dict(return_code=0, checked_episodes=len(rows),
            valid_reference_count=len(valid), summary_sha256=sha(out / 'summary.json'))
        save(mp, manifest)
        print('EXPERT_COMPLETE', len(valid), '/', len(selected), flush=True)
    except BaseException as error:
        manifest['status'] = manifest['units'][0]['status'] = 'failed'
        manifest['failure'] = repr(error)
        save(mp, manifest)
        save(out / 'exit.json', dict(return_code=1, error=repr(error)))
        raise
    finally:
        if controller is not None:
            controller.stop()


if __name__ == '__main__':
    main()
