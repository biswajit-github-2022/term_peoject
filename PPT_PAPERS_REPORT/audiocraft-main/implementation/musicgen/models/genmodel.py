

from abc import ABC, abstractmethod
import typing as tp

import omegaconf
import torch

from .encodec import CompressionModel
from .lm import LMModel
from .builders import get_wrapped_compression_model
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes
from ..utils.autocast import TorchAutocast


class BaseGenModel(ABC):
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: tp.Optional[float] = None):
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        self.cfg: tp.Optional[omegaconf.DictConfig] = None
                self.compression_model.eval()
        self.lm.eval()

        if hasattr(lm, 'cfg'):
            cfg = lm.cfg
            assert isinstance(cfg, omegaconf.DictConfig)
            self.cfg = cfg

        if self.cfg is not None:
            self.compression_model = get_wrapped_compression_model(self.compression_model, self.cfg)

        if max_duration is None:
            if self.cfg is not None:
                max_duration = lm.cfg.dataset.segment_duration              else:
                raise ValueError("You must provide max_duration when building directly your GenModel")
        assert max_duration is not None

        self.max_duration: float = max_duration
        self.duration = self.max_duration

                                self.extend_stride: tp.Optional[float] = None
        self.device = next(iter(lm.parameters())).device
        self.generation_params: dict = {}
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None
        if self.device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device.type, dtype=torch.float16)

    @property
    def frame_rate(self) -> float:
        return self.compression_model.sample_rate

    @property
    def audio_channels(self) -> int:
        self._progress_callback = progress_callback

    @abstractmethod
    def set_generation_params(self, *args, **kwargs):

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.

        Args:
            num_samples (int): Number of samples to be generated.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (list of str, optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (here text).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio
