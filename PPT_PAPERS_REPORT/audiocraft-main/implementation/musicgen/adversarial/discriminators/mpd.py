
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import NormConv2d
from .base import MultiDiscriminator, MultiDiscriminatorOutputType


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, in_channels: int = 1, out_channels: int = 1,
                 n_layers: int = 5, kernel_sizes: tp.List[int] = [5, 3], stride: int = 3,
                 filters: int = 8, filters_scale: int = 4, max_filters: int = 1024,
                 norm: str = 'weight_norm', activation: str = 'LeakyReLU',
                 activation_params: dict = {'negative_slope': 0.2}):
        super().__init__()
        self.period = period
        self.n_layers = n_layers
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.convs = nn.ModuleList()
        in_chs = in_channels
        for i in range(self.n_layers):
            out_chs = min(filters * (filters_scale ** (i + 1)), max_filters)
            eff_stride = 1 if i == self.n_layers - 1 else stride
            self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=(kernel_sizes[0], 1), stride=(eff_stride, 1),
                                         padding=((kernel_sizes[0] - 1) // 2, 0), norm=norm))
            in_chs = out_chs
        self.conv_post = NormConv2d(in_chs, out_channels, kernel_size=(kernel_sizes[1], 1), stride=1,
                                    padding=((kernel_sizes[1] - 1) // 2, 0), norm=norm)

    def forward(self, x: torch.Tensor):
        fmap = []
                b, c, t = x.shape
        if t % self.period != 0:              n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        
        return x, fmap


class MultiPeriodDiscriminator(MultiDiscriminator):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 periods: tp.Sequence[int] = [2, 3, 5, 7, 11], **kwargs):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p, in_channels, out_channels, **kwargs) for p in periods
        ])

    @property
    def num_discriminators(self):
        return len(self.discriminators)

    def forward(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps
