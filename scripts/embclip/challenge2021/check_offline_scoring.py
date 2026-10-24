"""Check public scoring inputs on the unmodified official Test build.

Evaluator only: no policy is loaded and no simulator restrictions are changed.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / args.manifest
    run = json.loads(manifest.read_text())
    out = root / run['result_path']
    if run['status'] != 'prepared' or sha(__file__) != run['runner_sha256']:
        raise RuntimeError('Unexpected source or experiment state')
    if out != Path(__file__).resolve().parent or (out / 'summary.json').exists():
        raise RuntimeError('Result leaf is not fresh')
    prior = json.loads((root / run['preflight_manifest']).read_text())
    runtime = json.loads((root / prior['runtime_receipt']).read_text())
    exe = root / runtime['executable']
    if sha(exe) != runtime['executable_sha256']:
        raise RuntimeError('Official simulator changed')
    import ai2thor
    from ai2thor.controller import Controller
    if ai2thor.__version__ != '2.7.2':
        raise RuntimeError('Wrong independent runtime')
    Controller.base_dir = property(lambda self: str(root / 'assets/embclip-robothor-challenge2021/ai2thor'))
    episodes = []
    for source in prior['test_data']:
        path = root / source['path']
        if sha(path) != source['sha256']:
            raise RuntimeError('Test data changed')
        episodes.append(json.load(gzip.open(path, 'rt'))[0])
    val = root / 'data/embclip-robothor/robothor-objectnav/val/episodes/FloorPlan_Val1_1.json.gz'
    episodes.insert(0, json.load(gzip.open(val, 'rt'))[0])
    controller = None
    run['status'] = run['units'][0]['status'] = 'running'
    save(manifest, run)
    try:
        config = prior['official_config']
        first = episodes[0]
        controller = Controller(local_executable_path=str(exe), scene=first['scene'],
            width=config['width'], height=config['height'], renderDepthImage=False,
            robothorChallengeEpisodeId=first['id'], **config['initialize'])
        checks = []
        for episode in episodes:
            controller.initialization_parameters['robothorChallengeEpisodeId'] = episode['id']
            controller.reset(episode['scene'])
            event = controller.step(action='TeleportFull', **episode['initial_position'],
                rotation=dict(x=0, y=episode['initial_orientation'], z=0),
                horizon=episode['initial_horizon'], standing=True)
            if not event.metadata['lastActionSuccess']:
                raise RuntimeError('Official episode initialization failed')
            objects = event.metadata.get('objects')
            row = dict(episode_id=episode['id'], scene=episode['scene'],
                dataset_has_shortest_path='shortest_path' in episode,
                metadata_keys=sorted(event.metadata), object_count=len(objects or []),
                target_metadata_count=sum(o.get('objectType') == episode['object_type'] for o in (objects or [])))
            for action in [dict(action='GetShortestPath', objectType=episode['object_type'],
                                position=episode['initial_position']), dict(action='Stop')]:
                event = controller.step(**action)
                row[action['action']] = dict(success=event.metadata['lastActionSuccess'],
                    error=event.metadata.get('errorMessage'), action_return=event.metadata.get('actionReturn'))
            checks.append(row)
            save(out / 'checks.json', checks)
            print(json.dumps(row), flush=True)
        summary = dict(status='complete', build_commit=runtime['build_commit'],
            checks=checks, policies_loaded=0, simulator_modified=False,
            official_scores_available=False)
        save(out / 'summary.json', summary)
        run['status'] = run['units'][0]['status'] = 'complete'
        run['completion_evidence'] = dict(return_code=0, checked_scenes=len(checks),
            summary_sha256=sha(out / 'summary.json'))
        save(manifest, run)
    except BaseException as error:
        run['status'] = run['units'][0]['status'] = 'failed'
        run['failure'] = repr(error)
        save(manifest, run)
        raise
    finally:
        if controller is not None:
            controller.stop()


if __name__ == '__main__':
    main()
