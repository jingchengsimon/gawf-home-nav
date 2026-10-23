"""Grad-CAM on pose-replayed successful validation views, with recorded provenance.

Frozen SJC result-leaf execution only. Pose metadata is used by the renderer;
models receive RGB/category and their own recurrent/feedback state only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/embclip-allenact')]


def main():
    import numpy as np
    import torch
    from allenact.base_abstractions.misc import Memory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', required=True)
    args = parser.parse_args()
    mp = ROOT / 'experiments/monitoring/jobs' / (args.experiment_id + '.json')
    m = json.loads(mp.read_text())
    out = ROOT / m['result_path']
    if m['status'] != 'prepared' or out != Path(__file__).resolve().parent:
        raise RuntimeError('Unexpected analysis identity')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != m['runner_sha256']:
        raise RuntimeError('Frozen runner changed')
    suite = ROOT / m['recording_suite']
    spec = importlib.util.spec_from_file_location('recorded_navigation', suite / 'runner.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    if helpers.sha(suite / 'runner.py') != m['recording_runner_sha256']:
        raise RuntimeError('Recording helpers changed')
    if json.loads((suite / 'exit.json').read_text())['return_code'] != 0:
        raise RuntimeError('Recording helpers have no successful exit receipt')
    source = json.loads((out / 'selection.json').read_text())
    if helpers.sha(out / 'selection.json') != m['selection_sha256']:
        raise RuntimeError('Selection changed')
    if m['comparison_mode'] not in ('shared_history', 'individual_terminal'):
        raise RuntimeError('Unspecified analysis protocol')
    for model, record in source['source_metrics'].items():
        summary = json.loads((ROOT/'results'/record['experiment_id']/'summary.json').read_text())
        metrics_path = ROOT / summary['metrics_path']
        if helpers.sha(metrics_path) != record['metrics_sha256']:
            raise RuntimeError('Validation metrics changed')
        tasks = {t['task_info']['id']: t for t in json.loads(metrics_path.read_text())[0]['tasks']}
        key = 'reference' if model == 'gawf1' else 'individual_baseline'
        for case in source['cases']:
            task = case[key]
            if tasks.get(task['task_info']['id']) != task:
                raise RuntimeError('Selected trajectory differs from verified validation metrics')
    torch.set_num_threads(1)
    bundles = {p['model']: helpers.load_bundle(p) for p in m['policies']}
    m['status'] = m['units'][0]['status'] = 'running'
    helpers.write(mp, m)
    records = []

    def capture(task_record, seed):
        """Re-render the stored validation poses, without passing poses to actors."""
        info = task_record['task_info']
        scene = info['scene']
        dp = ROOT / 'data/embclip-robothor/robothor-objectnav/val/episodes' / (scene + '.json.gz')
        expected_digest = m['validation_files'][str(dp.relative_to(ROOT))]
        episode = helpers.episode({'dataset_path': str(dp.relative_to(ROOT)),
                                  'dataset_sha256': expected_digest, 'episode_id': info['id']})
        if not task_record['success'] or info['mirrored'] or info['materials_randomized']:
            raise RuntimeError('Expected an unmodified successful validation task')
        actions = info['taken_actions']
        path = info['followed_path']
        if len(actions) != task_record['ep_length'] or len(path) < len(actions):
            raise RuntimeError('Validation action/pose coverage mismatch')
        if actions[-1] != 'End':
            raise RuntimeError('Successful validation task did not end with End')
        sampler, task = helpers.sampler_for(episode, seed)
        observations = []
        try:
            helpers.checked_action(task.env.controller, action='SetRandomSeed', seed=1000 + seed)
            for pose in path[:len(actions)]:
                helpers.checked_action(task.env.controller, action='TeleportFull',
                    x=pose['x'], y=pose['y'], z=pose['z'],
                    rotation=pose['rotation'], horizon=pose['horizon'], forceAction=True)
                actual = helpers.pose(task)
                if max(abs(actual['position'][k]-pose[k]) for k in ('x','y','z')) > 1e-3:
                    raise RuntimeError('Replayed pose differs from recorded validation pose')
                # Refresh AllenAct's event-backed observation; no task reward/info is used.
                obs = task.get_observations()
                if set(obs) != {'rgb_lowres', 'goal_object_type_ind'}:
                    raise RuntimeError('Policy observation boundary changed')
                observations.append({k: np.array(v).copy() for k, v in obs.items()})
            visible = any(o['objectType'] == info['object_type'] and o['visible']
                          for o in task.env.last_event.metadata['objects'])
            if not visible:
                raise RuntimeError('Replayed terminal target is not visible: ' + info['id'])
            return observations, [info['action_names'].index(a) for a in actions], episode
        finally:
            sampler.close()

    def attribute(name, observations, actions):
        model, graph = bundles[name]
        memory = helpers.memory_for(model)
        prev = torch.zeros((1, 1), dtype=torch.long, device='cuda:0')
        mask = torch.zeros((1, 1, 1), device='cuda:0')
        for i, obs in enumerate(observations):
            tensor = {k: torch.as_tensor(v, device='cuda:0').unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                processed = graph.get_observations(tensor)
            if i + 1 == len(observations):
                def saved_memory():
                    return Memory({k: (memory.tensor(k).clone(),
                        [d[0] for d in dims].index('sampler'))
                        for k, (dims, _) in model.recurrent_memory_specification.items()})
                feature = processed['rgb_clip_resnet'].detach().requires_grad_()
                processed['rgb_clip_resnet'] = feature
                # The legacy goal encoder reshapes entries in its input dictionary.
                # Give each forward its own dictionary, retaining the same feature leaf.
                def inputs():
                    return {k: v.unsqueeze(0) for k, v in processed.items()}
                with torch.no_grad():
                    ref, _ = model(inputs(), saved_memory(), prev, mask)
                with torch.backends.cudnn.flags(enabled=False):
                    result, _ = model(inputs(), saved_memory(), prev, mask)
                    delta = float((ref.distributions.logits - result.distributions.logits.detach()).abs().max())
                    if delta > 1e-3:
                        raise RuntimeError('Native backward forward mismatch')
                    score = result.distributions.logits[..., 3].sum()
                    grad = torch.autograd.grad(score, feature)[0]
                    cam = (feature * grad.mean((-2, -1), keepdim=True)).sum(1).relu()[0].detach().cpu().numpy()
                return cam, dict(gradient_l2=float(grad.norm()), native_forward_delta=delta,
                                 end_log_probability=float(score.detach()))
            with torch.no_grad():
                _, memory = model({k: v.unsqueeze(0) for k, v in processed.items()}, memory, prev, mask)
            prev.fill_(actions[i])
            mask.fill_(1)
        raise RuntimeError('Empty visual history')

    try:
        for index, case in enumerate(source['cases']):
            target = case['target']
            shared = None
            models = {}
            for name in ('gru1', 'gawf1'):
                reference = case['reference']
                if m['comparison_mode'] == 'individual_terminal' and name == 'gru1':
                    reference = case['individual_baseline']
                if m['comparison_mode'] == 'shared_history':
                    if shared is None:
                        shared = capture(reference, index)
                    observations, actions, episode = shared
                else:
                    observations, actions, episode = capture(reference, index)
                cam, diagnostic = attribute(name, observations, actions)
                means = np.array((.48145466, .4578275, .40821073))
                std = np.array((.26862954, .26130258, .27577711))
                rgb = np.clip(observations[-1]['rgb_lowres'] * std + means, 0, 1)
                np.save(out / (target + '_' + name + '_cam.npy'), cam)
                np.save(out / (target + '_' + name + '_rgb.npy'), rgb)
                rgb_hash = hashlib.sha256(observations[-1]['rgb_lowres'].tobytes()).hexdigest()
                history_hash = hashlib.sha256(b''.join(o['rgb_lowres'].tobytes() for o in observations)).hexdigest()
                models[name] = dict(episode_id=episode['id'], decision_step=len(observations)-1,
                    reference_actions=actions, terminal_rgb_sha256=rgb_hash,
                    history_rgb_sha256=history_hash, cam_sha256=helpers.sha(out / (target+'_'+name+'_cam.npy')),
                    rgb_sha256=helpers.sha(out / (target+'_'+name+'_rgb.npy')), **diagnostic)
            if m['comparison_mode'] == 'shared_history':
                if models['gru1']['history_rgb_sha256'] != models['gawf1']['history_rgb_sha256']:
                    raise RuntimeError('Shared observation comparison differs')
            records.append(dict(target=target, models=models))
            helpers.write(out/'progress.json', dict(completed_cases=len(records), expected_cases=12))
            print('COMPLETE', target, flush=True)
        if len(records) != 12 or len({r['target'] for r in records}) != 12:
            raise RuntimeError('Expected 12 unique target categories')
        summary = dict(status='complete', comparison_mode=m['comparison_mode'], records=records,
            method='End-action log-probability Grad-CAM at CLIP RN50 final convolutional features',
            rendering='Pose-replayed validation observations. Logged positions/orientations are evaluator-only and are never actor inputs. These are re-rendered histories, not claimed pixel-exact recordings of original evaluation.',
            history=('Each model receives its own pose-replayed successful validation RGB/action history and reconstructs its own recurrent state and feedback. Eleven categories share the same episode and initial pose; BaseballBat uses different successful episodes, explicitly marked.' if m['comparison_mode']=='individual_terminal' else 'Both models receive identical target-visible RGB sequences and reference prior actions, reset their own recurrent state, and generate their own internal policy feedback. This is a teacher-forced attribution diagnostic, not an autonomous navigation success comparison.'),
            normalization='Within-map positive maximum; identical rule for both models. Compare spatial distribution, not gradient brightness.',
            limits='Local gradient sensitivity, not intrinsic attention or a causal proof of navigation advantage.', weights_updated=False)
        helpers.write(out/'summary.json', summary)
        m['status'] = m['units'][0]['status'] = 'complete'
        m['completion_evidence'] = dict(return_code=0, summary_sha256=helpers.sha(out/'summary.json'))
        helpers.write(mp, m)
    except Exception as error:
        m['status'] = m['units'][0]['status'] = 'failed'
        m['failure'] = repr(error)
        helpers.write(mp, m)
        raise


if __name__ == '__main__':
    main()
