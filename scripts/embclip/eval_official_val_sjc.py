"""Evaluate the public 130,091,717-step EmbCLIP checkpoint on existing Val."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from scripts.embclip.eval_light5m_sjc import sha256, validation_ids, write_json, run_logged_evaluation
    manifest_path = root / args.manifest
    run = json.loads(manifest_path.read_text())
    out = root / run['units'][0]['result_path']
    if (run['status'] != 'prepared' or out != Path(__file__).resolve().parent
            or sha256(Path(__file__)) != run['runner_sha256']):
        raise RuntimeError('Unexpected manifest, source or result identity')
    if set(p.name for p in out.iterdir()) - {'runner.py', 'official_val_config.py', 'run.log'}:
        raise RuntimeError('Result leaf is not fresh')
    if sha256(out / 'official_val_config.py') != run['config_sha256']:
        raise RuntimeError('Frozen config changed')
    for name, digest in run['source']['source_files'].items():
        if sha256(root / name) != digest:
            raise RuntimeError('Evaluation source changed: ' + name)
    expected_ids, expected_scenes = validation_ids(run['validation']['files'])
    checkpoint = root / run['source_checkpoint']['path']
    if sha256(checkpoint) != run['source_checkpoint']['sha256']:
        raise RuntimeError('Official checkpoint changed')
    import torch
    ckpt = torch.load(checkpoint, map_location='cpu')
    if int(ckpt['total_steps']) != 130091717:
        raise RuntimeError('Wrong official checkpoint steps')
    del ckpt
    gpu = run['execution']['physical_gpu_index']
    uuid, free = subprocess.check_output(['nvidia-smi', '--id='+str(gpu),
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if int(free) < 11000 or any(s.split(',')[0].strip() == uuid.strip() for s in apps.splitlines()):
        raise RuntimeError('GPU occupied or insufficient memory')
    if shutil.disk_usage(out).free < 5 * 1024**3:
        raise RuntimeError('Insufficient result space')
    run['status'] = run['units'][0]['status'] = 'running'
    write_json(manifest_path, run)
    try:
        command = [str(root / '.envs/embclip-robothor/bin/python'), '-u',
            'vendor/embclip-allenact/allenact/main.py',
            str((out / 'official_val_config.py').relative_to(root)), '-b', '.', '--eval', '--seed', '0',
            '--config_kwargs', json.dumps(dict(headless=True, test_on_validation=True,
                test_gpu_ids=[0], num_test_processes=15)), '-c', str(checkpoint), '-o', str(out / 'allenact')]
        env = os.environ.copy()
        env.update(GAWF_HOME_NAV_ROOT=str(root), CUDA_VISIBLE_DEVICES=str(gpu),
            PATH=str(root / 'assets/embclip-robothor/vulkan-tools/package/usr/bin')+':'+env['PATH'],
            PYTHONPATH=str(root / 'vendor/embclip-allenact')+':'+str(root),
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        env.pop('DISPLAY', None)
        write_json(out / 'launch.json', dict(command=command, physical_gpu=gpu, evaluation_seed=0))
        rc = run_logged_evaluation(command, out, env, 1800)
        write_json(out / 'exit.json', dict(return_code=rc))
        if rc:
            raise RuntimeError('AllenAct evaluation exited '+str(rc))
        files = list((out / 'allenact/metrics').rglob('metrics__test_*.json'))
        if len(files) != 1:
            raise RuntimeError('Expected one final metrics file')
        metrics_path = files[0]
        rows = json.loads(metrics_path.read_text())
        if len(rows) != 1 or int(rows[0]['training_steps']) != 130091717:
            raise RuntimeError('Metrics checkpoint mismatch')
        tasks = rows[0]['tasks']
        ids = [t['task_info']['id'] for t in tasks]
        scenes = {t['task_info']['scene'] for t in tasks}
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= expected_ids or not scenes <= expected_scenes:
            raise RuntimeError('Invalid episode identities')
        sr = sum(bool(t['success']) for t in tasks) / len(tasks)
        spl = sum(float(t['spl']) for t in tasks) / len(tasks)
        if abs(sr-rows[0]['success']) > 1e-6 or abs(spl-rows[0]['spl']) > 1e-6:
            raise RuntimeError('Per-episode and aggregate metrics disagree')
        missing = sorted(expected_ids-set(ids))
        summary = dict(status='partial' if missing else 'complete', experiment_id=run['experiment_id'],
            episode_count=len(ids), unique_episode_count=len(ids), expected_episode_count=1800,
            missing_episode_ids=missing, scene_count=len(scenes), success_rate=sr, spl=spl,
            checkpoint_steps=130091717, checkpoint_sha256=sha256(checkpoint),
            metrics_path=str(metrics_path.relative_to(root)), metrics_sha256=sha256(metrics_path),
            ai2thor_version='5.0.0', ai2thor_commit='f0825767cd50d69f666c7f282e54abfe58f1e917', evaluation_seed=0)
        write_json(out / 'summary.json', summary)
        write_json(out / 'progress.json', dict(status=summary['status'], completed_episodes=len(ids), expected_episodes=1800))
        run['status'] = run['units'][0]['status'] = summary['status']
        run['completion_evidence'] = dict(return_code=0, episode_count=len(ids),
            summary_sha256=sha256(out / 'summary.json'), metrics_sha256=sha256(metrics_path))
        write_json(manifest_path, run)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        run['status'] = run['units'][0]['status'] = 'failed'
        run['failure'] = repr(error)
        write_json(manifest_path, run)
        raise
    finally:
        subprocess.run([sys.executable, '-B', 'scripts/rebuild_index.py'], cwd=root, check=True)


if __name__ == '__main__':
    main()
