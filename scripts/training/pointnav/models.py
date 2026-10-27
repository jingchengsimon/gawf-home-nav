"""Two-layer PointNav GaWF using the shared canonical recurrence.

Feedback is the preceding clean recurrent state of both layers. It is detached
at each timestep and masked at episode boundaries. The policy receives exactly
the same observations, visual encoder, action head, and critic as the LSTM.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from scripts.training.canonical_core.gawf import GaWFCore
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy


class GaWFStateEncoder(nn.Module):
    def __init__(self, input_size=576, hidden_size=512):
        super().__init__()
        self.core = GaWFCore(input_size, hidden_size, feedback_dim=2 * hidden_size,
                             num_layers=2, dropout=0.0, gate_tau=0.5)

    @property
    def num_recurrent_layers(self):
        return 2

    def _step(self, x, state, mask):
        state = state * mask.view(1, -1, 1)
        feedback = torch.cat((state[0], state[1]), dim=-1).detach()

        def run_core(input_t, first, second, feedback_t):
            output, next_layers = self.core.step_with_state(
                input_t, [first, second], [feedback_t, feedback_t]
            )
            return output, next_layers[0], next_layers[1]

        if self.training and torch.is_grad_enabled():
            output, first, second = checkpoint(
                run_core, x, state[0], state[1], feedback,
                preserve_rng_state=False, use_reentrant=False,
            )
        else:
            output, first, second = run_core(x, state[0], state[1], feedback)
        return output, torch.stack((first, second), dim=0)

    def forward(self, x, hidden_states, masks):
        if x.size(0) == hidden_states.size(1):
            return self._step(x, hidden_states, masks)
        count = hidden_states.size(1)
        if x.size(0) % count:
            raise ValueError("GaWF sequence length is not divisible by environment count")
        outputs = []
        for input_t, mask_t in zip(x.view(-1, count, x.size(-1)),
                                   masks.view(-1, count, 1)):
            output, hidden_states = self._step(input_t, hidden_states, mask_t)
            outputs.append(output)
        return torch.cat(outputs, dim=0), hidden_states


def make_gawf_policy(observation_space, action_space, ppo_cfg, ddppo_cfg, goal_sensor_uuid):
    policy = PointNavResNetPolicy(
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=ppo_cfg.hidden_size,
        rnn_type="LSTM",
        num_recurrent_layers=ddppo_cfg.num_recurrent_layers,
        backbone=ddppo_cfg.backbone,
        goal_sensor_uuid=goal_sensor_uuid,
        normalize_visual_inputs="rgb" in observation_space.spaces,
    )
    baseline = policy.net.state_encoder
    policy.net.state_encoder = GaWFStateEncoder(
        baseline.rnn.input_size, ppo_cfg.hidden_size
    )
    return policy
