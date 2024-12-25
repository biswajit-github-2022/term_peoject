
import typing as tp

import flashy
import julius
import omegaconf
import torch
import torch.nn.functional as F

from . import builders
from . import base
from .. import models
from ..modules.diffusion_schedule import NoiseSchedule
from ..metrics import RelativeVolumeMel
from ..models.builders import get_processor
from ..utils.samples.manager import SampleManager
from ..solvers.compression import CompressionSolver


class PerStageMetrics:
    def __init__(self, num_steps: int, num_stages: int = 4):
        self.num_steps = num_steps
        self.num_stages = num_stages

    def __call__(self, losses: dict, step: tp.Union[int, torch.Tensor]):
        if type(step) is int:
            stage = int((step / self.num_steps) * self.num_stages)
            return {f"{name}_{stage}": loss for name, loss in losses.items()}
        elif type(step) is torch.Tensor:
            stage_tensor = ((step / self.num_steps) * self.num_stages).long()
            out: tp.Dict[str, float] = {}
            for stage_idx in range(self.num_stages):
                mask = (stage_tensor == stage_idx)
                N = mask.sum()
                stage_out = {}
                if N > 0:                      for name, loss in losses.items():
                        stage_loss = (mask * loss).sum() / N
                        stage_out[f"{name}_{stage_idx}"] = stage_loss
                out = {**out, **stage_out}
            return out


class DataProcess:
    def __init__(self, initial_sr: int = 24000, target_sr: int = 16000, use_resampling: bool = False,
                 use_filter: bool = False, n_bands: int = 4,
                 idx_band: int = 0, device: torch.device = torch.device('cpu'), cutoffs=None, boost=False):
        assert idx_band < n_bands
        self.idx_band = idx_band
        if use_filter:
            if cutoffs is not None:
                self.filter = julius.SplitBands(sample_rate=initial_sr, cutoffs=cutoffs).to(device)
            else:
                self.filter = julius.SplitBands(sample_rate=initial_sr, n_bands=n_bands).to(device)
        self.use_filter = use_filter
        self.use_resampling = use_resampling
        self.target_sr = target_sr
        self.initial_sr = initial_sr
        self.boost = boost

    def process_data(self, x, metric=False):
        if x is None:
            return None
        if self.boost:
            x /= torch.clamp(x.std(dim=(1, 2), keepdim=True), min=1e-4)
            x * 0.22
        if self.use_filter and not metric:
            x = self.filter(x)[self.idx_band]
        if self.use_resampling:
            x = julius.resample_frac(x, old_sr=self.initial_sr, new_sr=self.target_sr)
        return x

    def inverse_process(self, x):

    The diffusion task allows for MultiBand diffusion model training.

    Args:
        cfg (DictConfig): Configuration.
        self.dataloaders = builders.get_audio_datasets(self.cfg)

    def show(self):
                raise NotImplementedError()

    def run_step(self, idx: int, batch: torch.Tensor, metrics: dict):
        Runs audio reconstruction evaluation.
        condition = self.get_condition(wav)
        initial = self.schedule.get_initial_noise(self.data_processor.process_data(wav))          result = self.schedule.generate_subsampled(self.model, initial=initial, condition=condition,
                                                   step_list=step_list)
        result = self.data_processor.inverse_process(result)
        return result

    def generate(self):
