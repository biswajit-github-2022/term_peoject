
import math
import typing as tp

import torch

from .base import BaseQuantizer, QuantizedResult
from .core_vq import ResidualVectorQuantization


class ResidualVectorQuantizer(BaseQuantizer):
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        q_dropout: bool = False,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: float = 2.,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = False,
        orthogonal_reg_max_codes: tp.Optional[int] = None,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            orthogonal_reg_weight=self.orthogonal_reg_weight,
            orthogonal_reg_active_codes_only=self.orthogonal_reg_active_codes_only,
            orthogonal_reg_max_codes=self.orthogonal_reg_max_codes,
            channels_last=False
        )

    def forward(self, x: torch.Tensor, frame_rate: int):
        n_q = self.n_q
        if self.training and self.q_dropout:
            n_q = int(torch.randint(1, self.n_q + 1, (1,)).item())
        bw_per_q = math.log2(self.bins) * frame_rate / 1000
        quantized, codes, commit_loss = self.vq(x, n_q=n_q)
        codes = codes.transpose(0, 1)
                bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        n_q = self.n_q
        codes = self.vq.encode(x, n_q=n_q)
        codes = codes.transpose(0, 1)
                return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
