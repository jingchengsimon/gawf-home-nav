"""Validate the official Test scenes and unchanged trained policies on SJC.

Metadata/poses are used only to initialize episodes and write submission traces.
Policy observations are the original RGB/category sensors and recurrent history.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import types


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, sort_keys=True) + '\n')


def agent_pose(event):
    a = event.metadata['agent']
    return dict(**a['position'], rotation=a['rotation']['y'], horizon=a['cameraHorizon'])


def reset_episode(controller, e):
    controller.initialization_parameters['robothorChallengeEpisodeId'] = e['id']
    controller.reset(e['scene'])
    event = controller.step(action='TeleportFull', **e['initial_position'],
        rotation=dict(x=0, y=e['initial_orientation'], z=0),
        horizon=e['initial_horizon'], standing=True)
    if not event.metadata['lastActionSuccess']:
        raise RuntimeError('Episode teleport failed: ' + e['id'])
    actual = agent_pose(event)
    if max(abs(actual[k] - e['initial_position'][k]) for k in ('x', 'y', 'z')) > 1e-3:
        raise RuntimeError('Initial position mismatch')
    if abs((actual['rotation']-e['initial_orientation']+180)%360-180)>1e-3 or abs(actual['horizon']-e['initial_horizon'])>1e-3:
        raise RuntimeError('Initial orientation mismatch')
    return event


def model_bundle(root, policy):
    from policy_adapter import model_bundle as load
    return load(root, policy)


def smoke_policy(root, policy, controller, e, out):
    import numpy as np
    import torch
    from allenact.base_abstractions.misc import Memory
    from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
    config, graph, model = model_bundle(root, policy)
    names = ObjectNavTask.class_action_names()
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    generator = torch.Generator(device='cuda:0').manual_seed(0)
    event = reset_episode(controller, e)
    initial_rgb = sha(out / (e['scene'] + '.png'))
    entries = {}
    for name,(dims,dtype) in model.recurrent_memory_specification.items():
        entries[name] = (torch.zeros(tuple(1 if s is None else s for _,s in dims),dtype=dtype,device='cuda:0'),[d[0] for d in dims].index('sampler'))
    memory = Memory(entries)
    prev = torch.zeros((1,1),dtype=torch.long,device='cuda:0')
    mask = torch.zeros((1,1,1),device='cuda:0')
    trajectory, actions = [agent_pose(event)], []
    start = time.monotonic()
    for _ in range(20):
        env = types.SimpleNamespace(current_frame=event.frame.copy())
        task = types.SimpleNamespace(task_info={'object_type':e['object_type']})
        observations = {s.uuid:s.get_observation(env=env,task=task) for s in config.SENSORS}
        tensors = {k:torch.as_tensor(v,device='cuda:0').unsqueeze(0) for k,v in observations.items()}
        with torch.no_grad():
            processed = graph.get_observations(tensors)
            if set(processed) != {'rgb_clip_resnet','goal_object_type_ind'}:
                raise RuntimeError('Unexpected policy input')
            result,memory = model({k:v.unsqueeze(0) for k,v in processed.items()},memory,prev,mask)
            action = int(torch.multinomial(result.distributions.probs.reshape(-1),1,generator=generator))
        name = names[action]
        event = controller.step(action='Stop' if name=='End' else name)
        actions.append(dict(action='Stop' if name=='End' else name, success=bool(event.metadata['lastActionSuccess'])))
        trajectory.append(agent_pose(event));prev.fill_(action);mask.fill_(1)
        if name == 'End':
            break
    receipt = dict(model=policy['model'],training_seed=policy['seed'],checkpoint_sha256=policy['checkpoint_sha256'],
        episode_id=e['id'],evaluation_seed=0,steps=len(actions),elapsed_seconds=time.monotonic()-start,
        trajectory=trajectory,actions_taken=actions,policy_inputs=['RGB','goal category','self-generated recurrent history'],
        metrics_available=False,preflight_only=True,max_preflight_steps=20,weights_updated=False,
        frozen_visual_weights_sha256=sha(root/'assets/embclip-robothor/home/.cache/clip/RN50.pt'))
    write(out/(policy['model']+'_policy_smoke.json'),receipt)
    del model,graph
    torch.cuda.empty_cache()
    return {k:v for k,v in receipt.items() if k not in ('trajectory','actions_taken')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    args=p.parse_args();root=args.root.resolve();mp=args.manifest.resolve();m=json.loads(mp.read_text());out=root/m['result_path']
    if m['status']!='prepared' or sha(__file__)!=m['runner_sha256'] or out!=Path(__file__).resolve().parent:
        raise RuntimeError('Unexpected manifest/source/result identity')
    if {x.name for x in out.iterdir()}-{'runner.py','run.log','policy_adapter.py','oracle.py','oracle.log','equivalence.json'}:
        raise RuntimeError('Result leaf is not fresh')
    if sha(out/'policy_adapter.py')!=m['adapter_sha256'] or sha(out/'oracle.py')!=m['equivalence_runner_sha256']:
        raise RuntimeError('Frozen adapter/oracle changed')
    equality=json.loads((out/'equivalence.json').read_text())
    if equality['status']!='pass' or equality['adapter_sha256']!=m['adapter_sha256']:
        raise RuntimeError('Original policy equivalence has not passed')
    for name,digest in m['source_files'].items():
        if sha(root/name)!=digest:raise RuntimeError('Policy source changed: '+name)
    sys.path[:0]=[str(root),str(root/'vendor/embclip-allenact')]
    import gzip
    import ai2thor
    import torch
    from ai2thor.controller import Controller
    from PIL import Image
    if ai2thor.__version__!='2.7.2' or torch.cuda.device_count()!=1:
        raise RuntimeError('Wrong independent client or visible GPU count')
    runtime=json.loads((root/m['runtime_receipt']).read_text());exe=root/runtime['executable']
    if sha(exe)!=runtime['executable_sha256']:
        raise RuntimeError('Official build executable changed')
    Controller.base_dir=property(lambda self:str(root/'assets/embclip-robothor-challenge2021/ai2thor'))
    episodes=[];scenes=[];policies=[];controller=None
    m['status']=m['units'][0]['status']='running';write(mp,m)
    try:
        for source in m['test_data']:
            path=root/source['path']
            if sha(path)!=source['sha256']:raise RuntimeError('Test dataset changed')
            items=json.load(gzip.open(path,'rt'));episodes.extend(items)
            if len(items)!=source['episodes'] or any('shortest_path_length' in x for x in items):raise RuntimeError('Unexpected Test records')
            scenes.append(items[0])
        if len(episodes)!=2040 or len({x['id'] for x in episodes})!=2040 or len(scenes)!=10:raise RuntimeError('Test coverage mismatch')
        for policy in m['all_seed_checkpoints']:
            path=root/policy['checkpoint_path']
            if sha(path)!=policy['checkpoint_sha256']:raise RuntimeError('Checkpoint hash mismatch')
            ckpt=torch.load(path,map_location='cpu')
            if int(ckpt['total_steps'])!=5001216 or int(ckpt['trainer_seed'])!=policy['seed']:raise RuntimeError('Seed/budget mismatch')
            del ckpt
        first=scenes[0]
        initialize=m['official_config']['initialize']
        controller=Controller(local_executable_path=str(exe),scene=first['scene'],
            width=m['official_config']['width'],height=m['official_config']['height'],
            robothorChallengeEpisodeId=first['id'],renderDepthImage=False,**initialize)
        checked=[]
        for e in scenes:
            event=reset_episode(controller,e)
            if event.frame.shape!=(480,640,3) or float(event.frame.std())<1:raise RuntimeError('Blank or wrong RGB frame')
            Image.fromarray(event.frame).save(out/(e['scene']+'.png'))
            changed=controller.step(action='RotateRight')
            if not changed.metadata['lastActionSuccess']:raise RuntimeError('Official action failed')
            checked.append(dict(scene=e['scene'],episode_id=e['id'],rgb_sha256=sha(out/(e['scene']+'.png'))))
            print('TEST_SCENE_PASS',e['scene'],flush=True)
        for policy in m['policies']:
            policies.append(smoke_policy(root,policy,controller,first,out))
            print('POLICY_SMOKE_PASS',policy['model'],policy['seed'],flush=True)
        summary=dict(status='complete',ai2thor_version=ai2thor.__version__,build_commit=runtime['build_commit'],
            executable_sha256=runtime['executable_sha256'],verified_test_episodes=2040,verified_scenes=checked,
            verified_seed_checkpoints=10,policy_smokes=policies,full_evaluation=False,official_scores_available=False,
            rendering='Existing user X11 display; no sudo or display permission changes.',
            compatibility='Architecture-only loading reuses the unchanged GaWF class AST and official sensors/CLIP preprocessor. Verified against original constructors; no vendor edits or simulator version spoofing.',
            policy_equivalence_sha256=sha(out/'equivalence.json'))
        write(out/'summary.json',summary);m['status']=m['units'][0]['status']='complete';m['completion_evidence']=dict(return_code=0,summary_sha256=sha(out/'summary.json'));write(mp,m)
    except BaseException as error:
        m['status']=m['units'][0]['status']='failed';m['failure']=repr(error);write(mp,m);write(out/'failure.json',dict(error=repr(error)));raise
    finally:
        if controller is not None:controller.stop()


if __name__=='__main__':main()
