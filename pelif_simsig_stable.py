"""Position-dependent soft-reset LIF neurons and the MPR-SG surrogate.

The equations and numerical stabilization match the released training code.
Thresholds are buffers rather than trainable parameters.
"""

import torch
from torch import nn
from spikingjelly.activation_based import neuron


def generate_thresh(num_rows, num_cols):
    """Return paired cosine/sine thresholds of shape [1, length, head_dim]."""
    if num_rows % 2:
        raise ValueError('The per-head embedding dimension must be even.')
    positions = torch.arange(num_cols).unsqueeze(1)
    channels = torch.arange(num_rows).unsqueeze(0)
    rates = 1.0 / torch.pow(10000, (2 * (channels // 2)) / num_rows)
    angles = positions * rates
    pattern = torch.zeros_like(angles)
    pattern[:, 0::2] = torch.cos(angles[:, 0::2])
    pattern[:, 1::2] = torch.sin(angles[:, 1::2])
    return pattern.unsqueeze(0)


class SimSigmoid(nn.Module):
    """Membrane Potential Regularization Surrogate Gradient (MPR-SG).

    The detached correction approximates hard spikes in the forward pass and
    differentiates the batch-mean membrane/spike discrepancy in the backward
    pass. The original alpha=4 and denominator epsilon are preserved.
    """

    def __init__(self, alpha=4.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, v, thr):
        difference = v - thr
        spikes = (difference >= 0).to(v)
        norm = torch.norm(spikes.mean(0).flatten() - v.mean(0).flatten(), p=2)
        smooth = (difference * self.alpha).sigmoid()
        return smooth + norm * ((spikes - smooth) / (norm.detach() + self.eps)).detach()


class PELIFNode(neuron.base.MemoryModule):
    """LIF with sinusoidal thresholds and soft reset.

    SpikingJelly's MemoryModule supplies the multi-step loop and reset(). Call
    functional.reset_net(model) between independent forward passes.
    """

    def __init__(self, v_threshold=1.0, tau=2.0, decay_input=True,
                 surrogate_function=None, detach_reset=True, step_mode='m',
                 num_heads=1, embedding_dim=768, token_num=128, k=0.3):
        super().__init__()
        if tau <= 1:
            raise ValueError('tau must be greater than one.')
        if embedding_dim % num_heads:
            raise ValueError('embedding_dim must be divisible by num_heads.')
        self.tau = tau
        self.decay_input = decay_input
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.base_v_threshold = float(v_threshold)
        self.surrogate_function = surrogate_function if surrogate_function is not None else SimSigmoid()
        self.register_memory('v', 0.)
        self.register_buffer('threshold_pattern',
                             generate_thresh(embedding_dim // num_heads, token_num).repeat(1, 1, num_heads),
                             persistent=False)
        self.register_buffer('v_threshold', None, persistent=False)
        self.pelif_k = None
        self.set_pelif_k(k)

    def set_pelif_k(self, k):
        self.pelif_k = float(k)
        self.v_threshold = self.base_v_threshold + self.pelif_k * self.threshold_pattern

    def single_step_forward(self, x):
        if x.shape[1] > self.v_threshold.shape[1]:
            raise ValueError('Sequence exceeds pelif_token_num; increase it in the model config.')
        if self.decay_input:
            self.v = self.v + (x - self.v) / self.tau
        else:
            self.v = self.v + (-self.v) / self.tau + x
        threshold = self.v_threshold[:, :x.shape[1], :]
        spikes = self.surrogate_function(self.v, threshold)
        self.v = self.v - threshold * (spikes.detach() if self.detach_reset else spikes)
        return spikes
