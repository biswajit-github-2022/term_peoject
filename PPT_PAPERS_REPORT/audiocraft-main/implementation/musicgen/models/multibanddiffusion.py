

import typing as tp

import torch
import julius

from .unet import DiffusionUnet
from ..modules.diffusion_schedule import NoiseSchedule
from .encodec import CompressionModel
from ..solvers.compression import CompressionSolver
from .loaders import load_compression_model, load_diffusion_models


class DiffusionProcess:
    def __init__(self, model: DiffusionUnet, noise_schedule: NoiseSchedule) -> None:
        self.model = model
        self.schedule = noise_schedule

    def generate(self, condition: torch.Tensor, initial_noise: torch.Tensor,
                 step_list: tp.Optional[tp.List[int]] = None):
        return self.schedule.generate_subsampled(model=self.model, initial=initial_noise, step_list=step_list,
                                                 condition=condition)


class MultiBandDiffusion:
    def __init__(self, DPs: tp.List[DiffusionProcess], codec_model: CompressionModel) -> None:
        self.DPs = DPs
        self.codec_model = codec_model
        self.device = next(self.codec_model.parameters()).device

    @property
    def sample_rate(self) -> int:
        return self.codec_model.sample_rate

    @staticmethod
    def get_mbd_musicgen(device=None):

        Args:
            bw (float): Bandwidth of the compression model.
            device (torch.device or str, optional): Device on which the models are loaded.
            n_q (int, optional): Number of quantizers to use within the compression model.
        Args:
            wav (torch.Tensor): The audio that we want to extract the conditioning from.
        Args:
        Args:
            emb (torch.Tensor): Conditioning embeddings
            size (None, torch.Size): Size of the output
                if None this is computed from the typical upsampling of the model.
            step_list (list[int], optional): list of Markov chain steps, defaults to 50 linearly spaced step.
        Args:
            wav (torch.Tensor): Audio to equalize.
            ref (torch.Tensor): Reference audio from which we match the spectrogram.
            n_bands (int): Number of bands of the eq.
            strictness (float): How strict the matching. 0 is no matching, 1 is exact matching.
        Args:
            wav (torch.Tensor): Original 'ground truth' audio.
            sample_rate (int): Sample rate of the input (and output) wav.
        Args:
            tokens (torch.Tensor): Discrete codes.
            n_bands (int): Bands for the eq matching.
