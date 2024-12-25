
import typing as tp

import numpy as np
from torchaudio.transforms import MelSpectrogram
import torch
from torch import nn
from torch.nn import functional as F

from ..modules import pad_for_conv1d


class MelSpectrogramWrapper(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: tp.Optional[int] = None,
                 n_mels: int = 80, sample_rate: float = 22050, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        hop_length = int(hop_length)
        self.hop_length = hop_length
        self.mel_transform = MelSpectrogram(n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                            win_length=win_length, f_min=f_min, f_max=f_max, normalized=normalized,
                                            window_fn=torch.hann_window, center=False)
        self.floor_level = floor_level
        self.log = log

    def forward(self, x):
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (p, p), "reflect")
                                x = pad_for_conv1d(x, self.n_fft, self.hop_length)
        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        B, C, freqs, frame = mel_spec.shape
        if self.log:
            mel_spec = torch.log10(self.floor_level + mel_spec)
        return mel_spec.reshape(B, C * freqs, frame)


class MelSpectrogramL1Loss(torch.nn.Module):
    def __init__(self, sample_rate: int, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
                 n_mels: int = 80, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.melspec = MelSpectrogramWrapper(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                             n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                             log=log, normalized=normalized, floor_level=floor_level)

    def forward(self, x, y):
        self.melspec.to(x.device)
        s_x = self.melspec(x)
        s_y = self.melspec(y)
        return self.l1(s_x, s_y)


class MultiScaleMelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate: int, range_start: int = 6, range_end: int = 11,
                 n_mels: int = 64, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 normalized: bool = False, alphas: bool = True, floor_level: float = 1e-5):
        super().__init__()
        l1s = list()
        l2s = list()
        self.alphas = list()
        self.total = 0
        self.normalized = normalized
        for i in range(range_start, range_end):
            l1s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=False, normalized=normalized, floor_level=floor_level))
            l2s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=True, normalized=normalized, floor_level=floor_level))
            if alphas:
                self.alphas.append(np.sqrt(2 ** i - 1))
            else:
                self.alphas.append(1)
            self.total += self.alphas[-1] + 1

        self.l1s = nn.ModuleList(l1s)
        self.l2s = nn.ModuleList(l2s)

    def forward(self, x, y):
        loss = 0.0
        self.l1s.to(x.device)
        self.l2s.to(x.device)
        for i in range(len(self.alphas)):
            s_x_1 = self.l1s[i](x)
            s_y_1 = self.l1s[i](y)
            s_x_2 = self.l2s[i](x)
            s_y_2 = self.l2s[i](y)
            loss += F.l1_loss(s_x_1, s_y_1) + self.alphas[i] * F.mse_loss(s_x_2, s_y_2)
        if self.normalized:
            loss = loss / self.total
        return loss
