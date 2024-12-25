

import typing as tp

import torch

from .encodec import CompressionModel
from .genmodel import BaseGenModel
from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model
from .loaders import load_compression_model, load_lm_model


class AudioGen(BaseGenModel):
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: tp.Optional[float] = None):
        super().__init__(name, compression_model, lm, max_duration)
        self.set_generation_params(duration=5)  
    @staticmethod
    def get_pretrained(name: str = 'facebook/audiogen-medium', device=None):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
                        compression_model = get_debug_compression_model(device, sample_rate=16000)
            lm = get_debug_lm_model(device)
            return AudioGen(name, compression_model, lm, max_duration=10)

        compression_model = load_compression_model(name, device=device)
        lm = load_lm_model(name, device=device)
        assert 'self_wav' not in lm.condition_provider.conditioners, \
            "AudioGen do not support waveform conditioning for now"
        return AudioGen(name, compression_model, lm)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 10.0, cfg_coef: float = 3.0,
                              two_step_cfg: bool = False, extend_stride: float = 2):
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
        }
