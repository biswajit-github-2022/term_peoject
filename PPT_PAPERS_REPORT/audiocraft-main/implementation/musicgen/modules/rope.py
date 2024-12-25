
import typing as tp

from torch import nn
import torch


class XPos(nn.Module):
    def __init__(self, dim: int, smoothing: float = 0.4, base_scale: int = 512,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert dim % 2 == 0
        assert dtype in [torch.float64, torch.float32]
        self.dtype = dtype
        self.base_scale = base_scale

        half_dim = dim // 2
        adim = torch.arange(half_dim, device=device, dtype=dtype)
        decay_rates = (adim / half_dim + smoothing) / (1.0 + smoothing)
        self.register_buffer("decay_rates", decay_rates)
        self.decay: tp.Optional[torch.Tensor] = None

    def get_decay(self, start: int, end: int):

    Args:
        dim (int): Embedding dimension (twice the number of frequencies).
        max_period (float): Maximum period of the rotation frequencies.
        xpos (bool): Use xPos, applies an exponential decay to rotation matrix.
        scale (float): Scale of positional embedding, set to 0 to deactivate.
        device (torch.device, optional): Device on which to initialize the module.
        dtype (torch.dtype): dtype to use to generate the embedding.
        if self.rotation is None or end > self.rotation.shape[0]:
            assert isinstance(self.frequencies, torch.Tensor)              idx = torch.arange(end, device=self.frequencies.device, dtype=self.dtype)
            angles = torch.outer(idx, self.frequencies)
            self.rotation = torch.polar(torch.ones_like(angles), angles)
        return self.rotation[start:end]

    def rotate(self, x: torch.Tensor, start: int = 0, time_dim: int = 1, invert_decay: bool = False):
        Supports streaming mode, in which query and key are not expected to have the same shape.
        In streaming mode, key will be of length [P + C] with P the cached past timesteps, but
        query will be [C] (typically C == 1).

        Args:
            query (torch.Tensor): Query to rotate.
            key (torch.Tensor): Key to rotate.
            start (int): Start index of the sequence for time offset.
            time_dim (int): which dimension represent the time steps.
