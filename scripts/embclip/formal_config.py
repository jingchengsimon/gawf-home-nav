"""Local experiment adapters for the EmbCLIP 200M-step comparison.

The frozen AllenAct vendor supplies the task, CLIP preprocessing, PPO, and
single-layer baseline. Only model depth/width, the planned budget, and the
global 60-environment distribution are changed here.
"""

import gym
import torch
from torch.utils.checkpoint import checkpoint
from torch.optim.lr_scheduler import LambdaLR

from allenact.base_abstractions.experiment_config import MachineParams
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.utils.experiment_utils import Builder, LinearDecay
from allenact_plugins.navigation_plugin.objectnav.models import ResnetTensorNavActorCritic
from projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig,
)

from scripts.embclip.canonical_core.gawf import GaWFCore
from scripts.embclip.sjc_config import DATA


TOTAL_STEPS = 200_000_000
GLOBAL_ENVIRONMENTS = 60
GPUS_PER_NODE = 4
NODES_PER_RUN = 4
GPUS_PER_RUN = GPUS_PER_NODE * NODES_PER_RUN


class ThreeLayerGaWFActorCritic(ResnetTensorNavActorCritic):
    """Official EmbCLIP visual/goal encoder with canonical three-layer GaWF.

    The previous raw policy logits are the only feedback. They are detached,
    masked at episode reset, and recomputed under current parameters during
    each PPO sequence forward. No reward or simulator metadata enters feedback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_rnn_layers=3, **kwargs)
        self.state_encoders = torch.nn.ModuleDict(
            {
                "single_belief": GaWFCore(
                    input_size=self.goal_visual_encoder.output_dims,
                    hidden_size=self._hidden_size,
                    feedback_dim=self.action_space.n,
                    num_layers=3,
                    dropout=0.0,
                    gate_tau=0.5,
                )
            }
        )

    @property
    def num_recurrent_layers(self):
        return 3

    def _recurrent_memory_specification(self):
        spec = super()._recurrent_memory_specification()
        spec["policy_feedback"] = (
            (("sampler", None), ("feedback", self.action_space.n)),
            torch.float32,
        )
        return spec

    def forward(self, observations, memory, prev_actions, masks):
        # This frozen EmbCLIP baseline has add_prev_actions=False.
        obs_embeds = self.forward_encoder(observations)
        state = memory.tensor("single_belief")
        feedback = memory.tensor("policy_feedback")
        core = self.state_encoders["single_belief"]
        beliefs = []
        raw_logits = []
        for step in range(obs_embeds.shape[0]):
            mask = masks[step].reshape(-1, 1)
            state = state * mask.unsqueeze(0)
            feedback = feedback * mask
            if self.training and torch.is_grad_enabled():
                # Retain recurrent states but recompute the canonical gates
                # during backward; a 128-step PPO sequence otherwise retains
                # every dense gate and exceeds A6000 memory.
                def run_core(x, h0, h1, h2, fb):
                    output, next_layers = core.step_with_state(
                        x, [h0, h1, h2], [fb] * 3
                    )
                    return (output, *next_layers)

                belief, *next_state = checkpoint(
                    run_core,
                    obs_embeds[step],
                    *state.unbind(0),
                    feedback,
                    preserve_rng_state=False,
                )
            else:
                belief, next_state = core.step_with_state(
                    obs_embeds[step], list(state.unbind(0)), [feedback] * 3
                )
            state = torch.stack(next_state, dim=0)
            logits = self.actor.linear(belief)
            beliefs.append(belief)
            raw_logits.append(logits)
            feedback = logits.detach()
        belief_sequence = torch.stack(beliefs)
        logits_sequence = torch.stack(raw_logits)
        memory.set_tensor("single_belief", state)
        memory.set_tensor("policy_feedback", feedback.detach())
        return (
            ActorCriticOutput(
                distributions=CategoricalDistr(logits=logits_sequence),
                values=self.critic(belief_sequence),
                extras={},
            ),
            memory,
        )


class FormalEmbCLIPConfig(ObjectNavRoboThorClipRGBPPOExperimentConfig):
    """Shared task and training settings for three paired model families."""

    MODEL = "gru1"
    HIDDEN_SIZE = 512

    TRAIN_DATASET_DIR = str(DATA / "train")
    VAL_DATASET_DIR = str(DATA / "val")
    TEST_DATASET_DIR = None

    def create_model(self, **kwargs):
        model_class = (
            ThreeLayerGaWFActorCritic if self.MODEL == "gawf3" else ResnetTensorNavActorCritic
        )
        return model_class(
            action_space=gym.spaces.Discrete(self.ACTION_SPACE.n),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=self.SENSORS[1].uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet",
            depth_resnet_preprocessor_uuid=None,
            hidden_size=self.HIDDEN_SIZE,
            num_rnn_layers=1 if self.MODEL == "gru1" else 3,
            goal_dims=32,
        ) if self.MODEL != "gawf3" else model_class(
            action_space=gym.spaces.Discrete(self.ACTION_SPACE.n),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=self.SENSORS[1].uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet",
            depth_resnet_preprocessor_uuid=None,
            hidden_size=self.HIDDEN_SIZE,
            goal_dims=32,
        )

    def training_pipeline(self, **kwargs):
        pipeline = super().training_pipeline(**kwargs)
        pipeline.pipeline_stages[0].max_stage_steps = TOTAL_STEPS
        pipeline.lr_scheduler_builder = Builder(
            LambdaLR, {"lr_lambda": LinearDecay(steps=TOTAL_STEPS)}
        )
        return pipeline

    def machine_params(self, mode="train", **kwargs):
        if mode == "valid":
            return MachineParams(nprocesses=0, devices=[-1])
        params = super().machine_params(mode=mode, **kwargs)
        if mode != "train":
            return params
        assert self.num_train_processes == GLOBAL_ENVIRONMENTS
        assert len(params.devices) == GPUS_PER_NODE
        assert len(params.sampler_devices) == GPUS_PER_NODE
        # 60 total samplers over 16 trainers: twelve with 4 and four with 3.
        global_processes = (4,) * 12 + (3,) * 4
        machine_id = kwargs.get("machine_id")
        if machine_id is not None:
            assert 0 <= machine_id < NODES_PER_RUN
            local_ids = list(
                range(machine_id * GPUS_PER_NODE, (machine_id + 1) * GPUS_PER_NODE)
            )
        else:
            local_ids = None
        return MachineParams(
            nprocesses=global_processes,
            devices=params.devices * NODES_PER_RUN,
            sampler_devices=params.sampler_devices * NODES_PER_RUN,
            sensor_preprocessor_graph=params.sensor_preprocessor_graph,
            local_worker_ids=local_ids,
        )
