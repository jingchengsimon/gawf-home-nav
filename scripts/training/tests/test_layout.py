"""Offline checks for consolidated imports, campaign coverage and config contracts."""

import ast
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from scripts.training.pointnav import run as pointnav_run

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class TrainingLayoutChecks(unittest.TestCase):
    def test_internal_imports_and_script_paths_exist(self):
        for path in (ROOT / 'training').rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith('scripts.'):
                        self.assertFalse(node.module.startswith('scripts.analysis.'), (path, node.module))
                        target = REPO / node.module.replace('.', '/')
                        self.assertTrue(target.is_dir() or target.with_suffix('.py').is_file(),
                                        (path, node.lineno, node.module))
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if value.startswith('scripts/') and value.endswith('.py'):
                        self.assertTrue((REPO / value).is_file(), (path, value))


    def test_pointnav_launcher_resolves_repository_root(self):
        self.assertEqual(pointnav_run.ROOT, REPO)
        self.assertTrue((pointnav_run.ROOT / 'configs/pointnav2020/light_rgb/task.yaml').is_file())


    def test_pointnav_runtime_enforces_rgb_and_fixed_goal(self):
        class Goal:
            def __init__(self, value):
                self.value = value

            def copy(self):
                return Goal(self.value)

            def __eq__(self, other):
                return SimpleNamespace(all=lambda: self.value == other.value)

        class Environment:
            def reset(self):
                return self.observations

            def step(self):
                return self.observations, 0.0, False, {}

        dependencies = {
            'scripts.training.pointnav.compat': module('compat', install=lambda root: None),
            'numpy': module('numpy'),
            'habitat_baselines.common.baseline_registry': module('registry', baseline_registry=SimpleNamespace(register_env=lambda **kwargs: lambda cls: cls)),
            'habitat_baselines.common.environments': module('env', NavRLEnv=Environment),
            'habitat_baselines.config.default': module('config', get_config=lambda *args: None),
        }
        spec = importlib.util.spec_from_file_location('isolated_runtime', ROOT / 'training/pointnav/runtime.py')
        runtime = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependencies), patch('os.chdir'), patch.object(sys, 'path', list(sys.path)):
            spec.loader.exec_module(runtime)
        environment = runtime.CheckedRGBPointNav2020()
        environment.observations = dict(rgb='image', pointgoal=Goal(1))
        environment.reset()
        environment.step()
        environment.observations['pointgoal'] = Goal(2)
        with self.assertRaises(AssertionError):
            environment.step()
        environment.observations['depth'] = 'forbidden'
        with self.assertRaises(AssertionError):
            environment.reset()


    def test_unified_embclip_config_keeps_model_and_pipeline_settings(self):
        class BaseConfig:
            ADVANCE_SCENE_ROLLOUT_PERIOD = 100
            ACTION_SPACE = SimpleNamespace(n=6)
            SENSORS = [None, SimpleNamespace(uuid='goal')]

            def __init__(self, **kwargs):
                self.num_train_processes = kwargs.get('num_train_processes', 16)
                self.train_gpu_ids = kwargs.get('train_gpu_ids', [0])

        actor = lambda **kwargs: kwargs
        pipeline_calls = []

        def pipeline(**kwargs):
            pipeline_calls.append(kwargs)
            return SimpleNamespace(pipeline_stages=[SimpleNamespace(max_stage_steps=300_000_000)])

        dependencies = {
            'gym': module('gym', spaces=SimpleNamespace(Discrete=lambda n: n)),
            'allenact.base_abstractions.experiment_config': module('config', MachineParams=actor),
            'allenact_plugins.navigation_plugin.objectnav.models': module('models', ResnetTensorNavActorCritic=actor),
            'projects.objectnav_baselines.mixins': module('mixins', ObjectNavPPOMixin=SimpleNamespace(training_pipeline=pipeline)),
            'projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo': module('base', ObjectNavRoboThorClipRGBPPOExperimentConfig=BaseConfig),
            'scripts.training.embclip.models': module('gawf', SingleLayerGaWFActorCritic=actor),
        }
        spec = importlib.util.spec_from_file_location('isolated_config', ROOT / 'training/embclip/config.py')
        config = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependencies):
            spec.loader.exec_module(config)
            for family, width in (('gru1', 512), ('gawf1', 1160)):
                instance = config.EmbCLIPConfig(model=family)
                result = instance.create_model(sensor_preprocessor_graph=SimpleNamespace(observation_spaces={}))
                self.assertEqual(result['hidden_size'], width)
                self.assertEqual(result.get('num_rnn_layers', 1), 1)
                self.assertEqual(instance.training_pipeline().pipeline_stages[0].max_stage_steps, 5_000_000)
                self.assertEqual(pipeline_calls[-1]['save_interval'], 100_000)
            with self.assertRaises(ValueError):
                config.EmbCLIPConfig(model='gru3')
            with self.assertRaises(ValueError):
                config.EmbCLIPConfig(model='gawf1', official=True)



if __name__ == "__main__":
    unittest.main()
