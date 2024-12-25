
from abc import ABC, abstractmethod
import typing as tp

import torch
import torch.nn as nn


FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
MultiDiscriminatorOutputType = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


class MultiDiscriminator(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
        ...

    @property
    @abstractmethod
    def num_discriminators(self) -> int:
        ...
