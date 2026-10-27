"""Canonical RNN, GRU, and LSTM cores.

The Elman RNN carries clean ReLU activity and drops only inter-layer transmissions:
`h_t = relu(layer_norm(W_ih x_t + W_hh h_(t-1) + b_ih + b_hh))`,
`out_t = dropout(h_t)` only for non-final layers; the top output is clean.
Its four affine parameters retain the historical ``nn.RNN`` names for checkpoint compatibility,
but no native ``nn.RNN`` forward or implicit activation is present.
GRU and LSTM delegate to native PyTorch stacks with their built-in inter-layer dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT_PROTOCOL = "inter_layer_only_v4"


def _remap_legacy_native_layers(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, object],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Load historical per-layer GRU/LSTM weights into a native stacked module."""

    del local_metadata, strict, missing_keys, unexpected_keys
    for layer_idx in range(module.num_layers):
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            old_key = f"{prefix}rnns.{layer_idx}.{name}_l0"
            new_key = f"{prefix}rnn.{name}_l{layer_idx}"
            if old_key in state_dict:
                if new_key in state_dict:
                    error_msgs.append(
                        f"Conflicting recurrent checkpoint keys: {old_key}, {new_key}"
                    )
                else:
                    state_dict[new_key] = state_dict.pop(old_key)


class ElmanAffine(nn.Module):
    """Checkpoint-compatible parameters for one activation-free Elman affine transform."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.weight_ih_l0 = nn.Parameter(torch.empty(self.hidden_size, self.input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(self.hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match the default initialization used by ``nn.RNN``."""

        bound = self.hidden_size**-0.5
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)


class RNNCore(nn.Module):
    """Ungated GaWF-matched recurrence with ReLU activity inside the loop."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        self.num_layers = int(num_layers)
        self.effective_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.dropout_protocol = DROPOUT_PROTOCOL

        if self.num_layers == 1:
            self.rnn = ElmanAffine(self.input_size, self.hidden_size)
            self.norm = nn.LayerNorm(self.hidden_size)
        else:
            layer_inputs = [self.input_size] + [self.hidden_size] * (self.num_layers - 1)
            self.rnns = nn.ModuleList(
                [ElmanAffine(size, self.hidden_size) for size in layer_inputs]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(num_layers)])

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a PyTorch-shaped zero state `(layers, batch, hidden)`."""

        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype
        )

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one timestep and return `(top_output, next_layer_states)`."""

        if state.ndim == 2 and self.num_layers == 1:
            state = state.unsqueeze(0)
        expected = (self.num_layers, x_t.size(0), self.hidden_size)
        if tuple(state.shape) != expected:
            raise ValueError(f"Expected RNN state {expected}, got {tuple(state.shape)}")

        layer_input = x_t
        next_states: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            if self.num_layers == 1:
                rnn, norm = self.rnn, self.norm
            else:
                rnn, norm = self.rnns[layer_idx], self.norms[layer_idx]
            preactivation = F.linear(layer_input, rnn.weight_ih_l0, rnn.bias_ih_l0) + F.linear(
                state[layer_idx], rnn.weight_hh_l0, rnn.bias_hh_l0
            )
            hidden = F.relu(norm(preactivation))
            next_states.append(hidden)
            layer_input = hidden
            if self.dropout != 0 and layer_idx < self.num_layers - 1:
                layer_input = F.dropout(
                    layer_input, p=self.dropout, training=self.training, inplace=False
                )
        return layer_input, torch.stack(next_states, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the batch-first sequence through the fixed in-loop ReLU recurrence."""

        if state is None:
            state = self.initial_state(x.size(0), x.device, x.dtype)
        outputs: list[torch.Tensor] = []
        for time_idx in range(x.size(1)):
            output, state = self.step(x[:, time_idx, :], state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state


class GRUCore(nn.Module):
    """Native stacked GRU with dropout only between recurrent layers."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        self.effective_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.dropout_protocol = DROPOUT_PROTOCOL
        self.rnn = nn.GRU(
            self.input_size, self.hidden_size, num_layers=self.num_layers,
            batch_first=True, dropout=self.effective_dropout,
        )
        # PyTorch 2.0 exposes only the underscored pre-hook registration API.
        self._register_load_state_dict_pre_hook(_remap_legacy_native_layers, with_module=True)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the native GRU sequence path."""

        return self.rnn(x, state)


class LSTMCore(nn.Module):
    """Native stacked LSTM with dropout only between recurrent layers."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        self.effective_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.dropout_protocol = DROPOUT_PROTOCOL
        self.rnn = nn.LSTM(
            self.input_size, self.hidden_size, num_layers=self.num_layers,
            batch_first=True, dropout=self.effective_dropout,
        )
        self._register_load_state_dict_pre_hook(_remap_legacy_native_layers, with_module=True)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the native LSTM sequence path."""

        return self.rnn(x, state)
