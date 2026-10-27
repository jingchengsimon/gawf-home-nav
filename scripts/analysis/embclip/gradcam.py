"""Candidate Grad-CAM on pose-replayed successful validation terminal views.

Frozen SJC result-leaf execution only. Pose metadata is used by the renderer;
models receive RGB/category and their own recurrent/feedback state only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import copy
import shutil

ROOT = (Path(__file__).resolve().parents[3] if Path(__file__).resolve().parent.name == "embclip"
        else Path(__file__).resolve().parents[2])
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/embclip-allenact')]


def replay_main():
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
            prefix = case.get('candidate_id', target)
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
                np.save(out / (prefix + '_' + name + '_cam.npy'), cam)
                np.save(out / (prefix + '_' + name + '_rgb.npy'), rgb)
                rgb_hash = hashlib.sha256(observations[-1]['rgb_lowres'].tobytes()).hexdigest()
                history_hash = hashlib.sha256(b''.join(o['rgb_lowres'].tobytes() for o in observations)).hexdigest()
                models[name] = dict(episode_id=episode['id'], decision_step=len(observations)-1,
                    reference_actions=actions, terminal_rgb_sha256=rgb_hash,
                    history_rgb_sha256=history_hash, cam_sha256=helpers.sha(out / (prefix+'_'+name+'_cam.npy')),
                    rgb_sha256=helpers.sha(out / (prefix+'_'+name+'_rgb.npy')), **diagnostic)
            if m['comparison_mode'] == 'shared_history':
                if models['gru1']['history_rgb_sha256'] != models['gawf1']['history_rgb_sha256']:
                    raise RuntimeError('Shared observation comparison differs')
            records.append(dict(target=target, candidate_id=prefix, same_successful_episode=case.get('same_successful_episode'), models=models))
            helpers.write(out/'progress.json', dict(completed_cases=len(records), expected_cases=len(source['cases'])))
            print('COMPLETE', prefix, flush=True)
        if not any('candidate_id' in case for case in source['cases']) and len(records) != 12:
            raise RuntimeError('Expected twelve target categories')
        if len(records) != len(source['cases']) or len({r['candidate_id'] for r in records}) != len(source['cases']):
            raise RuntimeError('Candidate coverage mismatch')
        summary = dict(status='complete', comparison_mode=m['comparison_mode'], records=records, selection_rule=source.get('selection_rule', 'Twelve target categories'),
            method='End-action log-probability Grad-CAM at CLIP RN50 final convolutional features',
            rendering='Pose-replayed validation observations. Logged positions/orientations are evaluator-only and are never actor inputs. These are re-rendered histories, not claimed pixel-exact recordings of original evaluation.',
            history=('Each model receives its own pose-replayed successful validation RGB/action history and reconstructs its own recurrent state and feedback. Shared successful episodes are preferred; candidate receipts explicitly identify different episodes.' if m['comparison_mode']=='individual_terminal' else 'Both models receive identical target-visible RGB sequences and reference prior actions, reset their own recurrent state, and generate their own internal policy feedback. This is a teacher-forced attribution diagnostic, not an autonomous navigation success comparison.'),
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


def sample_main():
    import numpy as np
    import torch
    from allenact.base_abstractions.misc import Memory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', required=True)
    args = parser.parse_args()
    mp = ROOT / 'experiments/monitoring/jobs' / (args.experiment_id + '.json')
    m = json.loads(mp.read_text());out = ROOT / m['result_path']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if m['status'] != 'prepared' or out != Path(__file__).resolve().parent or sha(Path(__file__)) != m['runner_sha256']:
        raise RuntimeError('Frozen sample identity mismatch')
    suite = ROOT / m['recording_suite']
    if sha(suite / 'runner.py') != m['recording_runner_sha256'] or json.loads((suite / 'exit.json').read_text())['return_code'] != 0:
        raise RuntimeError('Unverified recording helper')
    spec = importlib.util.spec_from_file_location('recording_helpers', suite / 'runner.py')
    helpers = importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    original = ROOT / m['baseline_analysis']
    if sha(original / 'summary.json') != m['baseline_summary_sha256'] or json.loads((original / 'exit.json').read_text())['return_code'] != 0:
        raise RuntimeError('Unverified baseline maps')
    target = m.get('target', 'Vase')
    baseline_prefix = m.get('baseline_prefix', target)
    baseline = next(r['models']['gru1'] for r in json.loads((original / 'summary.json').read_text())['records'] if r['target'] == target and (not m.get('baseline_candidate_id') or r.get('candidate_id') == m['baseline_candidate_id']))
    policy = next(p for p in m['policies'] if p['model'] == 'gawf1')
    torch.set_num_threads(1);model, graph = helpers.load_bundle(policy)
    m['status'] = m['units'][0]['status'] = 'running';helpers.write(mp, m)
    records, attempts = [], []
    for case in m['cases']:
        e = helpers.episode(case);seed = case['action_sampling_seed']
        prefix = e['id'] + '_rollout' + str(seed)
        helpers.set_seed(seed);sampler, task = helpers.sampler_for(e, seed)
        actions, trajectory = [], []
        try:
            helpers.checked_action(task.env.controller, action='SetRandomSeed', seed=1000 + seed)
            start = helpers.pose(task);mem = helpers.memory_for(model)
            prev = torch.zeros((1, 1), dtype=torch.long, device='cuda:0');mask = torch.zeros((1, 1, 1), device='cuda:0')
            generator = torch.Generator(device='cuda:0').manual_seed(seed)
            while not task.is_done():
                obs = task.get_observations()
                if set(obs) != {'rgb_lowres', 'goal_object_type_ind'}:raise RuntimeError('Actor boundary changed')
                tensors = {k: torch.as_tensor(v, device='cuda:0').unsqueeze(0) for k, v in obs.items()}
                before = {k: mem.tensor(k).detach().clone() for k in model.recurrent_memory_specification}
                def memory():
                    return Memory({k: (before[k].clone(), [d[0] for d in dims].index('sampler')) for k, (dims, _) in model.recurrent_memory_specification.items()})
                with torch.no_grad():
                    processed = graph.get_observations(tensors)
                    result, next_mem = model({k: v.unsqueeze(0) for k, v in processed.items()}, mem, prev, mask)
                    action = int(torch.multinomial(result.distributions.probs.reshape(-1), 1, generator=generator).item())
                trajectory.append(helpers.pose(task));actions.append(task.class_action_names()[action]);task.step(action)
                if task.is_done():
                    metric = task.metrics();success = bool(metric['success'])
                    visible = any(o['objectType'] == e['object_type'] and o['visible'] for o in task.env.last_event.metadata['objects'])
                    attempt = dict(candidate_id=prefix, episode_id=e['id'], action_sampling_seed=seed, simulator_seed=1000+seed, training_seed=policy['seed'], checkpoint_sha256=policy['checkpoint_sha256'], steps=len(actions), success=success, target_visible=visible, start_actual=start, actions=actions, trajectory=trajectory, spl=float(metric['spl']))
                    attempts.append(attempt);helpers.write(out / (prefix + '.json'), attempt)
                    if success and visible and action == 3:
                        probe = out / (prefix + '_terminal_probe.pt')
                        torch.save(dict(rgb=np.array(obs['rgb_lowres']), goal=np.array(obs['goal_object_type_ind']), memory={k: v.cpu() for k, v in before.items()}, prev_actions=prev.cpu(), masks=mask.cpu(), action=action, success=True, target_visible=True, decision_step=len(actions)-1), probe)
                        feature = processed['rgb_clip_resnet'].detach().requires_grad_();processed['rgb_clip_resnet'] = feature
                        inputs = lambda: {k: v.unsqueeze(0) for k, v in processed.items()}
                        with torch.backends.cudnn.flags(enabled=False):
                            diagnostic, _ = model(inputs(), memory(), prev, mask)
                            delta = float((result.distributions.logits - diagnostic.distributions.logits.detach()).abs().max())
                            if delta > 1e-3:raise RuntimeError('Native forward mismatch')
                            score = diagnostic.distributions.logits[..., 3].sum();grad = torch.autograd.grad(score, feature)[0]
                            cam = (feature * grad.mean((-2, -1), keepdim=True)).sum(1).relu()[0].detach().cpu().numpy()
                        rgb = np.clip(obs['rgb_lowres'] * np.array((.26862954, .26130258, .27577711)) + np.array((.48145466, .4578275, .40821073)), 0, 1)
                        np.save(out / (prefix + '_gawf1_cam.npy'), cam);np.save(out / (prefix + '_gawf1_rgb.npy'), rgb)
                        models = {'gru1': dict(baseline), 'gawf1': dict(episode_id=e['id'], decision_step=len(actions)-1, source_receipt=str((out/(prefix+'.json')).relative_to(ROOT)), source_receipt_sha256=sha(out/(prefix+'.json')), terminal_probe_sha256=sha(probe), action_sampling_seed=seed, simulator_seed=1000+seed, native_forward_delta=delta, gradient_l2=float(grad.norm()), end_log_probability=float(score.detach()))}
                        for name in ('gru1', 'gawf1'):
                            for kind in ('cam', 'rgb'):
                                dest = out / (prefix + '_' + name + '_' + kind + '.npy')
                                if name == 'gru1':
                                    source = original / (baseline_prefix + '_gru1_' + kind + '.npy')
                                    if sha(source) != baseline[kind+'_sha256']:raise RuntimeError('Baseline changed')
                                    shutil.copyfile(source, dest)
                                models[name][kind+'_sha256'] = sha(dest)
                        records.append(dict(target=target, candidate_id=prefix, same_successful_episode=baseline['episode_id']==e['id'], models=models))
                    print('FINISHED', prefix, len(actions), success, flush=True)
                    break
                mem = next_mem;prev.fill_(action);mask.fill_(1)
        finally:
            sampler.close()
        helpers.write(out/'progress.json', dict(completed_attempts=len(attempts), expected_attempts=len(m['cases']), successful_cases=len(records)))
    if len(attempts) != len(m['cases']):raise RuntimeError('Rollout attempt coverage mismatch')
    summary = dict(status='complete', records=records, attempts=attempts, comparison_mode='individual_terminal', method='End-action log-probability Grad-CAM at CLIP RN50 final convolutional features', rendering='GaWF actual RGB and pre-End recurrent/feedback state from fixed-seed autonomous validation rollouts; baseline retained from verified successful pose-replayed terminal view.', normalization='Within-map positive maximum; identical rule for both models.', weights_updated=False)
    helpers.write(out/'summary.json', summary);m['status'] = m['units'][0]['status'] = 'complete';m['completion_evidence'] = dict(return_code=0, summary_sha256=sha(out/'summary.json'));helpers.write(mp,m)



def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def select_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Output must be a new leaf')
    selected = json.loads(args.selection.read_text())
    summaries = {}
    leaves = {'original': args.original, str(args.candidates): args.candidates}
    for replacement in selected['replacements'].values():
        leaf = Path(replacement.get('analysis', str(args.candidates)))
        leaves[str(leaf)] = leaf
    for name, leaf in leaves.items():
        summary = json.loads((leaf / 'summary.json').read_text())
        if summary['status'] != 'complete' or json.loads((leaf / 'exit.json').read_text())['return_code'] != 0:
            raise RuntimeError('Source analysis has no verified successful exit')
        summaries[name] = summary
    records = []
    plan = []
    for old in summaries['original']['records']:
        target = old['target']
        if target in selected['replacements']:
            identity = selected['replacements'][target]['candidate_id']
            chosen_leaf = Path(selected['replacements'][target].get('analysis', str(args.candidates)))
            matches = [r for r in summaries[str(chosen_leaf)]['records'] if r['candidate_id'] == identity and r['target'] == target]
            if len(matches) != 1:
                raise RuntimeError('Candidate identity is not unique')
            record, leaf, prefix = copy.deepcopy(matches[0]), chosen_leaf, identity
        else:
            record, leaf, prefix = copy.deepcopy(old), args.original, target
        record['source_analysis'] = leaf.name
        record['source_summary_sha256'] = digest(leaf / 'summary.json')
        for model in ('gru1', 'gawf1'):
            for kind in ('cam', 'rgb'):
                source = leaf / (prefix + '_' + model + '_' + kind + '.npy')
                if digest(source) != record['models'][model][kind + '_sha256']:
                    raise RuntimeError('Selected source array changed')
                plan.append((source, target + '_' + model + '_' + kind + '.npy'))
        records.append(record)
    if len(records) != 12 or len({r['target'] for r in records}) != 12:
        raise RuntimeError('Expected twelve unique categories')
    args.output.mkdir(parents=True)
    for source, name in plan:
        shutil.copyfile(source, args.output / name)
    summary = dict(status='complete', records=records, comparison_mode='individual_terminal',
                   assembler_sha256=digest(Path(__file__)),
                   method=summaries['original']['method'],
                   rendering='Each model uses its own successful terminal RGB and reconstructed or actually saved recurrent state, as specified by the source analysis and per-record provenance. Original validation pose replays and fixed-seed autonomous validation samples are both explicitly identified.',
                   normalization=summaries['original']['normalization'], weights_updated=False,
                   selection=selected, selection_sha256=digest(args.selection),
                   limits='Illustrative examples selected partly for target visibility and target-aligned genuine End-action Grad-CAM; not a localization benchmark or a causal proof of navigation advantage.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'selection.json').write_bytes(args.selection.read_bytes())
    (args.output / 'exit.json').write_text('{"return_code": 0}\n')
    print('Assembled 48 unchanged source arrays for twelve categories:', args.output)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=('gradcam-replay', 'gradcam-sample', 'select'))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    {'gradcam-replay': replay_main, 'gradcam-sample': sample_main, 'select': select_main}[args.operation]()


if __name__ == "__main__":
    main()
