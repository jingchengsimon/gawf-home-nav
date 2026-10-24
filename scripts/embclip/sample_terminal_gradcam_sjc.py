"""Sample fixed-seed validation rollouts; retain genuine successful terminal Grad-CAM."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
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
    baseline = next(r['models']['gru1'] for r in json.loads((original / 'summary.json').read_text())['records'] if r['target'] == 'Vase')
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
                                    source = original / ('Vase_gru1_' + kind + '.npy')
                                    if sha(source) != baseline[kind+'_sha256']:raise RuntimeError('Baseline changed')
                                    shutil.copyfile(source, dest)
                                models[name][kind+'_sha256'] = sha(dest)
                        records.append(dict(target='Vase', candidate_id=prefix, same_successful_episode=baseline['episode_id']==e['id'], models=models))
                    print('FINISHED', prefix, len(actions), success, flush=True)
                    break
                mem = next_mem;prev.fill_(action);mask.fill_(1)
        finally:
            sampler.close()
        helpers.write(out/'progress.json', dict(completed_attempts=len(attempts), expected_attempts=len(m['cases']), successful_cases=len(records)))
    if len(attempts) != len(m['cases']):raise RuntimeError('Rollout attempt coverage mismatch')
    summary = dict(status='complete', records=records, attempts=attempts, comparison_mode='individual_terminal', method='End-action log-probability Grad-CAM at CLIP RN50 final convolutional features', rendering='GaWF actual RGB and pre-End recurrent/feedback state from fixed-seed autonomous validation rollouts; baseline retained from verified successful pose-replayed terminal view.', normalization='Within-map positive maximum; identical rule for both models.', weights_updated=False)
    helpers.write(out/'summary.json', summary);m['status'] = m['units'][0]['status'] = 'complete';m['completion_evidence'] = dict(return_code=0, summary_sha256=sha(out/'summary.json'));helpers.write(mp,m)


if __name__ == '__main__':
    main()
