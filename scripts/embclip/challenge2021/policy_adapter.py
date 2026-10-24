"""Architecture-only loading for the official legacy simulator client.

No current simulator configuration class is imported. The GaWF class AST is
executed unchanged from its hash-checked training source. Equivalence against
both original 5M configuration constructors is a required SJC gate.
"""
import ast
from pathlib import Path
import types


def target_types(root):
    path = root / 'vendor/embclip-allenact/projects/objectnav_baselines/experiments/robothor/objectnav_robothor_base.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ObjectNavRoboThorBaseConfig')
    assignment = next(n for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TARGET_TYPES' for t in n.targets))
    # The original expression is tuple(sorted([literal category names])).
    names = ast.literal_eval(assignment.value.args[0].args[0])
    if len(names)!=12 or len(set(names))!=12 or not all(isinstance(n,str) for n in names):
        raise RuntimeError('Unexpected original category vocabulary')
    return tuple(sorted(names))


def gawf_class(root):
    import torch
    from torch.utils.checkpoint import checkpoint
    from allenact.base_abstractions.distributions import CategoricalDistr
    from allenact.base_abstractions.misc import ActorCriticOutput
    from allenact_plugins.navigation_plugin.objectnav.models import ResnetTensorNavActorCritic
    from scripts.embclip.canonical_core.gawf import GaWFCore
    path = root / 'scripts/embclip/light_gawf1_config.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SingleLayerGaWFActorCritic')
    namespace=dict(torch=torch,checkpoint=checkpoint,CategoricalDistr=CategoricalDistr,
        ActorCriticOutput=ActorCriticOutput,ResnetTensorNavActorCritic=ResnetTensorNavActorCritic,
        GaWFCore=GaWFCore,__name__=__name__)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['SingleLayerGaWFActorCritic']


def model_bundle(root, policy):
    import torch
    import gym
    import clip
    from allenact.base_abstractions.sensor import SensorSuite
    from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
    from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor,GoalObjectTypeThorSensor
    from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
    from allenact_plugins.navigation_plugin.objectnav.models import ResnetTensorNavActorCritic
    from projects.objectnav_baselines.experiments.clip.mixins import ClipResNetPreprocessGRUActorCriticMixin
    cache=root/'assets/embclip-robothor/home/.cache/clip'
    if not (cache/'RN50.pt').is_file():raise RuntimeError('Existing frozen RN50 weights missing')
    if not getattr(clip.load,'_challenge_cache',False):
        original=clip.load
        def load(*args,**kwargs):
            kwargs.setdefault('download_root',str(cache))
            return original(*args,**kwargs)
        load._challenge_cache=True
        clip.load=load
    sensors=[RGBSensorThor(height=224,width=224,use_resnet_normalization=True,uuid='rgb_lowres',
        mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,stdev=ClipResNetPreprocessor.CLIP_RGB_STDS),
        GoalObjectTypeThorSensor(object_types=target_types(root))]
    mixin=ClipResNetPreprocessGRUActorCriticMixin(sensors=sensors,clip_model_type='RN50',screen_size=224,goal_sensor_type=GoalObjectTypeThorSensor)
    graph=SensorPreprocessorGraph(source_observation_spaces=SensorSuite(sensors).observation_spaces,
        preprocessors=mixin.preprocessors()).to(torch.device('cuda:0'))
    common=dict(action_space=gym.spaces.Discrete(6),observation_space=graph.observation_spaces,
        goal_sensor_uuid=sensors[1].uuid,rgb_resnet_preprocessor_uuid='rgb_clip_resnet',
        depth_resnet_preprocessor_uuid=None,goal_dims=32)
    if policy['model']=='gru1':
        model=ResnetTensorNavActorCritic(hidden_size=512,num_rnn_layers=1,rnn_type='GRU',**common)
    elif policy['model']=='gawf1':
        model=gawf_class(root)(hidden_size=1160,**common)
    else:raise RuntimeError('Unsupported policy family')
    state=torch.load(root/policy['checkpoint_path'],map_location='cpu')
    if int(state['total_steps'])!=5001216 or int(state['trainer_seed'])!=policy['seed']:
        raise RuntimeError('Checkpoint seed/budget mismatch')
    model.load_state_dict(state['model_state_dict'],strict=True)
    return types.SimpleNamespace(SENSORS=sensors),graph,model.cuda().eval()
