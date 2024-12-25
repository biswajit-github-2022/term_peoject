

import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from audiocraft.models.loaders import load_audioseal_models


class WMModel(ABC, nn.Module):

    @abstractmethod
    def get_watermark(
        self,
        x: torch.Tensor,
        message: tp.Optional[torch.Tensor] = None,
        sample_rate: int = 16_000,
    ) -> torch.Tensor:

    @abstractmethod
    def detect_watermark(self, x: torch.Tensor) -> torch.Tensor:


class AudioSeal(WMModel):

    def __init__(
        self,
        generator: nn.Module,
        detector: nn.Module,
        nbits: int = 0,
    ):
        super().__init__()
        self.generator = generator          self.detector = detector  
                self.nbits = nbits if nbits else self.generator.msg_processor.nbits

    def get_watermark(
        self,
        x: torch.Tensor,
        message: tp.Optional[torch.Tensor] = None,
        sample_rate: int = 16_000,
    ) -> torch.Tensor:
        return self.generator.get_watermark(x, message=message, sample_rate=sample_rate)

    def detect_watermark(self, x: torch.Tensor) -> torch.Tensor:

                result = self.detector.detector(x)                  result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)
        return result

    def forward(          self,
        x: torch.Tensor,
        message: tp.Optional[torch.Tensor] = None,
        sample_rate: int = 16_000,
        alpha: float = 1.0,
    ) -> torch.Tensor:
