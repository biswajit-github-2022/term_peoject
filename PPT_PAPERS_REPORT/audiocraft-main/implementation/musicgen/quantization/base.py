

from dataclasses import dataclass, field
import typing as tp

import torch
from torch import nn


@dataclass
class QuantizedResult:
    x: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor      penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class BaseQuantizer(nn.Module):

    def forward(self, x: torch.Tensor, frame_rate: int) -> QuantizedResult:
        raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def total_codebooks(self):
        raise NotImplementedError()

    def set_num_codebooks(self, n: int):
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        return 1

    @property
    def num_codebooks(self):
        raise AttributeError("Cannot override the number of codebooks for the dummy quantizer")
