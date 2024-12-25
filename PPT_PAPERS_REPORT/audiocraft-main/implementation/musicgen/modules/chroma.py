import typing as tp

from einops import rearrange
from librosa import filters
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio


class ChromaExtractor(nn.Module):
    def __init__(self, sample_rate: int, n_chroma: int = 12, radix2_exp: int = 12, nfft: tp.Optional[int] = None,
                 winlen: tp.Optional[int] = None, winhop: tp.Optional[int] = None, argmax: bool = False,
                 norm: float = torch.inf):
        super().__init__()
        self.winlen = winlen or 2 ** radix2_exp
        self.nfft = nfft or self.winlen
        self.winhop = winhop or (self.winlen // 4)
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma
        self.norm = norm
        self.argmax = argmax
        self.register_buffer('fbanks', torch.from_numpy(filters.chroma(sr=sample_rate, n_fft=self.nfft, tuning=0,
                                                                       n_chroma=self.n_chroma)), persistent=False)
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.nfft, win_length=self.winlen,
                                                      hop_length=self.winhop, power=2, center=True,
                                                      pad=0, normalized=True)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        T = wav.shape[-1]
                        if T < self.nfft:
            pad = self.nfft - T
            r = 0 if pad % 2 == 0 else 1
            wav = F.pad(wav, (pad // 2, pad // 2 + r), 'constant', 0)
            assert wav.shape[-1] == self.nfft, f"expected len {self.nfft} but got {wav.shape[-1]}"

        spec = self.spec(wav).squeeze(1)
        raw_chroma = torch.einsum('cf,...ft->...ct', self.fbanks, spec)
        norm_chroma = torch.nn.functional.normalize(raw_chroma, p=self.norm, dim=-2, eps=1e-6)
        norm_chroma = rearrange(norm_chroma, 'b d t -> b t d')

        if self.argmax:
            idx = norm_chroma.argmax(-1, keepdim=True)
            norm_chroma[:] = 0
            norm_chroma.scatter_(dim=-1, index=idx, value=1)

        return norm_chroma
