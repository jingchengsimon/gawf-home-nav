"""One-layer GaWF paired with the light GRU1 EmbCLIP baseline."""

import torch
from torch.utils.checkpoint import checkpoint

from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact_plugins.navigation_plugin.objectnav.models import ResnetTensorNavActorCritic

from scripts.training.canonical_core.gawf import GaWFCore


class SingleLayerGaWFActorCritic(ResnetTensorNavActorCritic):
    """Use previous detached raw policy logits as the sole GaWF feedback."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_rnn_layers=1, **kwargs)
        self.state_encoders = torch.nn.ModuleDict({
            "single_belief": GaWFCore(
                input_size=self.goal_visual_encoder.output_dims,
                hidden_size=self._hidden_size,
                feedback_dim=self.action_space.n,
                num_layers=1,
                dropout=0.0,
                gate_tau=0.5,
            )
        })

    @property
    def num_recurrent_layers(self):
        return 1

    def _recurrent_memory_specification(self):
        spec = super()._recurrent_memory_specification()
        spec["policy_feedback"] = (
            (("sampler", None), ("feedback", self.action_space.n)),
            torch.float32,
        )
        return spec

    def forward(self, observations, memory, prev_actions, masks):
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
                def run_core(x, h, fb):
                    return core.step_with_state(x, h, fb)

                belief, next_hidden = checkpoint(
                    run_core, obs_embeds[step], state[0], feedback,
                    preserve_rng_state=False,
                )
            else:
                belief, next_hidden = core.step_with_state(
                    obs_embeds[step], state[0], feedback
                )
            state = next_hidden.unsqueeze(0)
            logits = self.actor.linear(belief)
            beliefs.append(belief)
            raw_logits.append(logits)
            feedback = logits.detach()
        belief_sequence = torch.stack(beliefs)
        memory.set_tensor("single_belief", state)
        memory.set_tensor("policy_feedback", feedback.detach())
        return (
            ActorCriticOutput(
                distributions=CategoricalDistr(logits=torch.stack(raw_logits)),
                values=self.critic(belief_sequence),
                extras={},
            ),
            memory,
        )
