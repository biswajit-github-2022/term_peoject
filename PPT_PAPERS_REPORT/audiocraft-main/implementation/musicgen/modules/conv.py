
import math
import typing as tp
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm


CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_group_norm'])


def apply_parametrization_norm(module: nn.Module, norm: str = 'none'):
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
                        return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs):
    assert norm in CONV_NORMALIZATIONS
    if norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0           1   2   3                   0 0 1 2 3 4 5 0                 1 2 3 4             If this is the case, we insert extra 0 padding to the right before the reflection happen.
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]


class NormConv1d(nn.Module):
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConv2d(nn.Module):
    def __init__(self, *args, norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class NormConvTranspose2d(nn.Module):
    def __init__(self, *args, norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal=False, norm=norm, **norm_kwargs)

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class StreamableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect'):
        super().__init__()
                if stride > 1 and dilation > 1:
            warnings.warn("StreamableConv1d has been initialized with stride > 1 and dilation > 1"
                          f" (kernel_size={kernel_size} stride={stride}, dilation={dilation}).")
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, groups=groups, bias=bias, causal=causal,
                               norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1          padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
                        x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
                        padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class StreamableConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, causal: bool = False,
                 norm: str = 'none', trim_right_ratio: float = 1.,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                          causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

                                        if self.causal:
                                    padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
                        padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y
