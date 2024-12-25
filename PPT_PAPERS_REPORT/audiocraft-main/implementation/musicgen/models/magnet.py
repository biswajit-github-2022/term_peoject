
import typing as tp
import torch

from .genmodel import BaseGenModel
from .loaders import load_compression_model, load_lm_model_magnet


class MAGNeT(BaseGenModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                self.duration = self.lm.cfg.dataset.segment_duration
        self.set_generation_params()

    @staticmethod
    def get_pretrained(name: str = 'facebook/magnet-small-10secs', device=None):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        compression_model = load_compression_model(name, device=device)
        lm = load_lm_model_magnet(name, compression_model_frame_rate=int(compression_model.frame_rate), device=device)

        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        kwargs = {'name': name, 'compression_model': compression_model, 'lm': lm}
        return MAGNeT(**kwargs)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 0,
                              top_p: float = 0.9, temperature: float = 3.0,
                              max_cfg_coef: float = 10.0, min_cfg_coef: float = 1.0,
                              decoding_steps: tp.List[int] = [20, 10, 10, 10],
                              span_arrangement: str = 'nonoverlap'):
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'max_cfg_coef': max_cfg_coef,
            'min_cfg_coef': min_cfg_coef,
            'decoding_steps': [int(s) for s in decoding_steps],
            'span_arrangement': span_arrangement
        }
