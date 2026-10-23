"""Small SJC check for matched recurrent parameters, reset and PPO gradients."""

import torch

from pointnav2020_gawf import GaWFStateEncoder


def main():
    torch.manual_seed(17)
    reference = torch.nn.LSTM(576, 512, num_layers=2)
    encoder = GaWFStateEncoder(576, 512)
    baseline_parameters = sum(value.numel() for value in reference.parameters())
    gawf_parameters = sum(value.numel() for value in encoder.parameters())
    gap = abs(gawf_parameters - baseline_parameters) / baseline_parameters
    assert gap < 0.01, (baseline_parameters, gawf_parameters, gap)
    x = torch.randn(6, 2, 576, requires_grad=True)
    state = torch.randn(2, 2, 512)
    masks = torch.ones(6, 2, 1)
    masks[2, 0] = 0
    output, final = encoder(x.reshape(-1, 576), state, masks.reshape(-1, 1))
    assert output.shape == (12, 512) and final.shape == (2, 2, 512)
    loss = output.square().mean() + final.square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in encoder.parameters())
    with torch.no_grad():
        one = encoder(x[2], state, masks[2])[0]
        two = encoder(x[2], torch.randn_like(state), masks[2])[0]
        assert torch.allclose(one[0], two[0], atol=1e-6)
    print(dict(lstm_recurrent_parameters=baseline_parameters,
               gawf_recurrent_parameters=gawf_parameters,
               relative_parameter_gap=gap, loss=float(loss)), flush=True)


if __name__ == "__main__":
    main()
