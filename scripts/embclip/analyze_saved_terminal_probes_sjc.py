"""Compute unchanged End-action Grad-CAM from actual saved successful terminal states."""
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
        raise RuntimeError('Frozen analysis identity mismatch')
    suite = ROOT / m['recording_suite']
    if sha(suite / 'runner.py') != m['recording_runner_sha256'] or json.loads((suite / 'exit.json').read_text())['return_code'] != 0:
        raise RuntimeError('Unverified recording suite')
    spec = importlib.util.spec_from_file_location('recording_helpers', suite / 'runner.py')
    helpers = importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    original = ROOT / m['baseline_analysis']
    if sha(original / 'summary.json') != m['baseline_summary_sha256'] or json.loads((original / 'exit.json').read_text())['return_code'] != 0:
        raise RuntimeError('Unverified baseline maps')
    baseline = {r['target']: r['models']['gru1'] for r in json.loads((original / 'summary.json').read_text())['records']}
    torch.set_num_threads(1)
    model, graph = helpers.load_bundle(next(p for p in m['policies'] if p['model'] == 'gawf1'))
    m['status'] = m['units'][0]['status'] = 'running';helpers.write(mp, m)
    records = []
    for case in m['cases']:
        prefix, target = case['candidate_id'], case['target']
        receipt_path = ROOT / case['receipt'];receipt = json.loads(receipt_path.read_text())
        policy = next(p for p in m['policies'] if p['model'] == 'gawf1')
        if sha(receipt_path) != case['receipt_sha256'] or receipt['checkpoint_sha256'] != policy['checkpoint_sha256'] or receipt['training_seed'] != policy['seed']:
            raise RuntimeError('Probe policy or receipt changed')
        probe = receipt_path.with_name(receipt_path.stem + '_terminal_probe.pt')
        if sha(probe) != receipt['terminal_probe_sha256'] or not receipt['success']:
            raise RuntimeError('Unverified successful terminal probe')
        p = torch.load(probe, map_location='cpu')
        if not p['success'] or not p['target_visible'] or int(p['action']) != 3:
            raise RuntimeError('Expected target-visible successful End decision')
        def memory():
            return Memory({k: (p['memory'][k].cuda().clone(), [d[0] for d in dims].index('sampler')) for k, (dims, _) in model.recurrent_memory_specification.items()})
        obs = {'rgb_lowres': torch.as_tensor(p['rgb'], device='cuda:0').unsqueeze(0), 'goal_object_type_ind': torch.as_tensor(p['goal'], device='cuda:0').unsqueeze(0)}
        with torch.no_grad():processed = graph.get_observations(obs)
        feature = processed['rgb_clip_resnet'].detach().requires_grad_();processed['rgb_clip_resnet'] = feature
        inputs = lambda: {k: v.unsqueeze(0) for k, v in processed.items()}
        prev, mask = p['prev_actions'].cuda(), p['masks'].cuda()
        with torch.no_grad():ref, _ = model(inputs(), memory(), prev, mask)
        with torch.backends.cudnn.flags(enabled=False):
            result, _ = model(inputs(), memory(), prev, mask)
            delta = float((ref.distributions.logits - result.distributions.logits.detach()).abs().max())
            if delta > 1e-3:raise RuntimeError('Native forward mismatch')
            score = result.distributions.logits[..., 3].sum();grad = torch.autograd.grad(score, feature)[0]
            cam = (feature * grad.mean((-2, -1), keepdim=True)).sum(1).relu()[0].detach().cpu().numpy()
        rgb = np.clip(p['rgb'] * np.array((.26862954, .26130258, .27577711)) + np.array((.48145466, .4578275, .40821073)), 0, 1)
        np.save(out / (prefix + '_gawf1_cam.npy'), cam);np.save(out / (prefix + '_gawf1_rgb.npy'), rgb)
        models = {'gru1': dict(baseline[target]), 'gawf1': dict(episode_id=receipt['episode']['id'], decision_step=p['decision_step'], source_receipt=case['receipt'], source_receipt_sha256=case['receipt_sha256'], terminal_probe_sha256=sha(probe), action_sampling_seed=receipt['action_sampling_seed'], simulator_seed=receipt['simulator_seed'], gradient_l2=float(grad.norm()), native_forward_delta=delta, end_log_probability=float(score.detach()))}
        for name in ('gru1', 'gawf1'):
            for kind in ('cam', 'rgb'):
                dest = out / (prefix + '_' + name + '_' + kind + '.npy')
                if name == 'gru1':
                    source = original / (target + '_' + name + '_' + kind + '.npy')
                    if sha(source) != baseline[target][kind + '_sha256']:raise RuntimeError('Baseline array changed')
                    shutil.copyfile(source, dest)
                models[name][kind + '_sha256'] = sha(dest)
        records.append(dict(target=target, candidate_id=prefix, same_successful_episode=models['gru1']['episode_id']==models['gawf1']['episode_id'], models=models))
        print('COMPLETE', prefix, flush=True)
    summary = dict(status='complete', records=records, comparison_mode='individual_terminal', method='End-action log-probability Grad-CAM at CLIP RN50 final convolutional features', rendering='GaWF uses the actual RGB and recurrent/feedback state saved before a successful End action; baseline retains its verified successful pose-replayed terminal state.', normalization='Within-map positive maximum; identical rule for both models.', weights_updated=False)
    helpers.write(out / 'summary.json', summary)
    m['status'] = m['units'][0]['status'] = 'complete';m['completion_evidence'] = dict(return_code=0, summary_sha256=sha(out / 'summary.json'));helpers.write(mp, m)


if __name__ == '__main__':
    main()
