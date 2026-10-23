"""Execute exactly one registered, bounded preflight or DD-PPO training smoke."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
from pointnav2020_compat import install
install(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_id')
    args = parser.parse_args()
    if not re.fullmatch(r'nav-[a-z0-9]+(?:-[a-z0-9]+)+', args.experiment_id):
        parser.error('One complete experiment ID required')
    manifest = json.loads((ROOT / 'experiments/monitoring/jobs' / (args.experiment_id + '.json')).read_text())
    assert manifest['experiment_id'] == args.experiment_id and manifest['host'] in ('amarel', 'sjc-remote')
    for relative, digest in manifest['source_files'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, 'Frozen source changed: ' + relative
    mode = manifest['mode']
    gpus = manifest['resources']['gpus']
    updates = manifest['budget']['updates']
    timeout = manifest['budget']['runtime_seconds']
    assert mode in ('preflight', 'train-smoke') and 1 <= gpus <= 4
    assert 1 <= timeout <= 1800 and 1 <= updates <= 16
    results = ROOT / 'results' / args.experiment_id
    logs = ROOT / 'logs' / args.experiment_id
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    assert not (results / 'exit.json').exists(), 'Preserve the previous run; register a new attempt/version'
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'vendor/habitat-lab-2020') + os.pathsep + str(ROOT / 'scripts'),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', GLOO_SOCKET_IFNAME='lo', NCCL_SOCKET_IFNAME='lo',
                       MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet', GLOG_minloglevel='2')
    # torch.distributed.launch and the original DD-PPO trainer each create a
    # TCPStore. Give them separate rendezvous ports on this single node.
    with socket.socket() as launcher_socket, socket.socket() as trainer_socket:
        launcher_socket.bind(('127.0.0.1', 0))
        trainer_socket.bind(('127.0.0.1', 0))
        launch_port = launcher_socket.getsockname()[1]
        trainer_port = trainer_socket.getsockname()[1]
    environment.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(launch_port),
                       POINTNAV_DDPPO_MASTER_PORT=str(trainer_port))
    python = str(ROOT / '.envs/pointnav2020/bin/python')
    if mode == 'preflight':
        command = [python, '-u', str(ROOT / 'scripts/preflight_pointnav2020.py'), args.experiment_id]
    else:
        command = [python, '-u', '-m', 'torch.distributed.launch', '--use_env', '--nproc_per_node', str(gpus),
                   '--master_addr', '127.0.0.1', '--master_port', str(launch_port), str(ROOT / 'scripts/train_pointnav2020.py'),
                   args.experiment_id, '--updates', str(updates)]
    (results / 'command.json').write_text(json.dumps({'command': command, 'mode': mode, 'budget': manifest['budget']}, indent=2))
    started = time.monotonic()
    exit_code, error = None, None
    try:
        with (logs / 'worker.log').open('w') as output:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        exit_code = completed.returncode
        assert exit_code == 0, 'Worker exit code %s; inspect worker.log' % exit_code
        if mode == 'preflight':
            summary = json.loads((results / 'summary.json').read_text())
            assert summary['status'] == 'pass'
        else:
            import torch
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ranks = [json.loads((results / ('rank_%d.complete.json' % rank)).read_text()) for rank in range(gpus)]
            starts = [json.loads((results / ('rank_%d.started.json' % rank)).read_text()) for rank in range(gpus)]
            assert all(r['world_size'] == gpus and r['updates'] == updates and r['status'] == 'complete' for r in ranks)
            assert {r['local_rank'] for r in starts} == set(range(gpus))
            checkpoint_paths = sorted((results / 'checkpoints').glob('ckpt.*.pth'))
            assert len(checkpoint_paths) == updates
            checks, first, changed = [], None, False
            for path in checkpoint_paths:
                checkpoint = torch.load(path, map_location='cpu')
                state = checkpoint['state_dict']
                assert state and all(torch.isfinite(value).all() for value in state.values() if torch.is_tensor(value))
                if first is None:
                    first = {k: v.clone() for k, v in state.items() if torch.is_tensor(v)}
                else:
                    changed |= any(not torch.equal(v, first[k]) for k, v in state.items() if torch.is_tensor(v))
                checks.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'global_frames': checkpoint['extra_state']['step']})
            assert changed, 'No tensor changed after the first update'
            scalars = {}
            for folder in sorted({p.parent for p in (logs / 'tensorboard').rglob('events.out.tfevents.*')}):
                events = EventAccumulator(str(folder)).Reload()
                for tag in events.Tags()['scalars']:
                    values = events.Scalars(tag)
                    assert values and all(math.isfinite(v.value) for v in values)
                    scalars[str(folder.relative_to(logs / 'tensorboard')) + '/' + tag] = {'step': values[-1].step, 'value': values[-1].value}
            assert any('losses_value' in key for key in scalars) and any('losses_policy' in key for key in scalars)
            assert any('reward' in key for key in scalars)
            frames = checks[-1]['global_frames']
            assert frames > 0
            summary = {'status': 'pass', 'experiment_id': args.experiment_id, 'completed_updates': updates,
                       'world_size': gpus, 'global_frames': frames, 'ranks': ranks, 'devices': starts,
                       'global_frames_per_rank0_wall_second': frames / ranks[0]['elapsed_s'],
                       'checkpoints': checks, 'all_checkpoint_tensors_finite': True,
                       'parameters_changed': changed, 'last_scalars': scalars,
                       'timing_note': 'Global frames / rank0 wrapper time includes setup; legacy train.log fps uses the trainer window. env_time/pth_time are rank0-local.'}
        summary['driver_elapsed_s'] = time.monotonic() - started
        (results / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        (results / 'exit.json').write_text(json.dumps({'return_code': exit_code, 'error': error,
                                                      'driver_elapsed_s': time.monotonic() - started}, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
