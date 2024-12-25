

from collections import namedtuple
import random
import typing as tp
import julius
import torch

TrainingItem = namedtuple("TrainingItem", "noisy noise step")


def betas_from_alpha_bar(alpha_bar):
    alphas = torch.cat([torch.Tensor([alpha_bar[0]]), alpha_bar[1:]/alpha_bar[:-1]])
    return 1 - alphas


class SampleProcessor(torch.nn.Module):
    def project_sample(self, x: torch.Tensor):
        return z


class MultiBandProcessor(SampleProcessor):
    def __init__(self, n_bands: int = 8, sample_rate: float = 24_000,
                 num_samples: int = 10_000, power_std: tp.Union[float, tp.List[float], torch.Tensor] = 1.):
        super().__init__()
        self.n_bands = n_bands
        self.split_bands = julius.SplitBands(sample_rate, n_bands=n_bands)
        self.num_samples = num_samples
        self.power_std = power_std
        if isinstance(power_std, list):
            assert len(power_std) == n_bands
            power_std = torch.tensor(power_std)
        self.register_buffer('counts', torch.zeros(1))
        self.register_buffer('sum_x', torch.zeros(n_bands))
        self.register_buffer('sum_x2', torch.zeros(n_bands))
        self.register_buffer('sum_target_x2', torch.zeros(n_bands))
        self.counts: torch.Tensor
        self.sum_x: torch.Tensor
        self.sum_x2: torch.Tensor
        self.sum_target_x2: torch.Tensor

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        return mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        return std

    @property
    def target_std(self):
        target_std = self.sum_target_x2 / self.counts
        return target_std

    def project_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        bands = self.split_bands(x)
        if self.counts.item() < self.num_samples:
            ref_bands = self.split_bands(torch.randn_like(x))
            self.counts += len(x)
            self.sum_x += bands.mean(dim=(2, 3)).sum(dim=1)
            self.sum_x2 += bands.pow(2).mean(dim=(2, 3)).sum(dim=1)
            self.sum_target_x2 += ref_bands.pow(2).mean(dim=(2, 3)).sum(dim=1)
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std          bands = (bands - self.mean.view(-1, 1, 1, 1)) * rescale.view(-1, 1, 1, 1)
        return bands.sum(dim=0)

    def return_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        bands = self.split_bands(x)
        rescale = (self.std / self.target_std) ** self.power_std
        bands = bands * rescale.view(-1, 1, 1, 1) + self.mean.view(-1, 1, 1, 1)
        return bands.sum(dim=0)


class NoiseSchedule:
    def __init__(self, beta_t0: float = 1e-4, beta_t1: float = 0.02, num_steps: int = 1000, variance: str = 'beta',
                 clip: float = 5., rescale: float = 1., device='cuda', beta_exp: float = 1,
                 repartition: str = "power", alpha_sigmoid: dict = {}, n_bands: tp.Optional[int] = None,
                 sample_processor: SampleProcessor = SampleProcessor(), noise_scale: float = 1.0, **kwargs):

        self.beta_t0 = beta_t0
        self.beta_t1 = beta_t1
        self.variance = variance
        self.num_steps = num_steps
        self.clip = clip
        self.sample_processor = sample_processor
        self.rescale = rescale
        self.n_bands = n_bands
        self.noise_scale = noise_scale
        assert n_bands is None
        if repartition == "power":
            self.betas = torch.linspace(beta_t0 ** (1 / beta_exp), beta_t1 ** (1 / beta_exp), num_steps,
                                        device=device, dtype=torch.float) ** beta_exp
        else:
            raise RuntimeError('Not implemented')
        self.rng = random.Random(1234)

    def get_beta(self, step: tp.Union[int, torch.Tensor]):
        if self.n_bands is None:
            return self.betas[step]
        else:
            return self.betas[:, step]  
    def get_initial_noise(self, x: torch.Tensor):
        if self.n_bands is None:
            return torch.randn_like(x)
        return torch.randn((x.size(0), self.n_bands, x.size(2)))

    def get_alpha_bar(self, step: tp.Optional[tp.Union[int, torch.Tensor]] = None) -> torch.Tensor:

        Args:
            x (torch.Tensor): clean audio data torch.tensor(bs, 1, T)
            tensor_step (bool): If tensor_step = false, only one step t is sample,
                the whole batch is diffused to the same step and t is int.
                If tensor_step = true, t is a tensor of size (x.size(0),)
                every element of the batch is diffused to a independently sampled.

        Args:
            model (nn.Module): Diffusion model.
            initial (tensor): Initial Noise.
            condition (tensor): Input conditionning Tensor (e.g. encodec compressed representation).
            return_list (bool): Whether to return the whole process or only the sampled point.
        if step_list is None:
            step_list = list(range(1000))[::-50] + [0]
        alpha_bar = self.get_alpha_bar(step=self.num_steps - 1)
        alpha_bars_subsampled = (1 - self.betas).cumprod(dim=0)[list(reversed(step_list))].cpu()
        betas_subsampled = betas_from_alpha_bar(alpha_bars_subsampled)
        current = initial * self.noise_scale
        iterates = [current]
        for idx, step in enumerate(step_list[:-1]):
            with torch.no_grad():
                estimate = model(current, step, condition=condition).sample * self.noise_scale
            alpha = 1 - betas_subsampled[-1 - idx]
            previous = (current - (1 - alpha) / (1 - alpha_bar).sqrt() * estimate) / alpha.sqrt()
            previous_alpha_bar = self.get_alpha_bar(step_list[idx + 1])
            if step == step_list[-2]:
                sigma2 = 0
                previous_alpha_bar = torch.tensor(1.0)
            else:
                sigma2 = (1 - previous_alpha_bar) / (1 - alpha_bar) * (1 - alpha)
            if sigma2 > 0:
                previous += sigma2**0.5 * torch.randn_like(previous) * self.noise_scale
            if self.clip:
                previous = previous.clamp(-self.clip, self.clip)
            current = previous
            alpha_bar = previous_alpha_bar
            if step == 0:
                previous *= self.rescale
            if return_list:
                iterates.append(previous.cpu())
        if return_list:
            return iterates
        else:
            return self.sample_processor.return_sample(previous)
