
from omegaconf import DictConfig
from . import builders, musicgen
from einops import rearrange
from torch.nn import functional as F
from ..modules.conditioners import SegmentWithAttributes

import torch
import numpy as np
import random
import typing as tp
import math
import flashy


class MagnetSolver(musicgen.MusicGenSolver):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

                self.generation_params = {
            'use_sampling': self.cfg.generate.lm.use_sampling,
            'temp': self.cfg.generate.lm.temp,
            'top_k': self.cfg.generate.lm.top_k,
            'top_p': self.cfg.generate.lm.top_p,
            'max_cfg_coef': self.cfg.generate.lm.max_cfg_coef,
            'min_cfg_coef': self.cfg.generate.lm.min_cfg_coef,
            'decoding_steps': list(self.cfg.generate.lm.decoding_steps),
            'anneal_temp': self.cfg.generate.lm.anneal_temp,
            'span_scoring': self.cfg.generate.lm.span_scoring,
            'span_arrangement': self.cfg.generate.lm.span_arrangement
        }

        sequence_len = int(cfg.dataset.segment_duration * self.compression_model.frame_rate)
        self.mean_maskrate_to_u = torch.tensor(self._calc_mean_maskrate_to_u_LUT(sequence_len), device=self.device)
        self.ce_per_codebook = [torch.log(torch.tensor(self.compression_model.cardinality, device=self.device))
                                for _ in range(cfg.transformer_lm.n_q)]

    def build_model(self) -> None:
        self.cfg.transformer_lm.segment_duration = self.cfg.dataset.segment_duration
        self.cfg.transformer_lm.span_len = self.cfg.masking.span_len
        assert self.cfg.efficient_attention_backend == "xformers", "MAGNeT v1 models support only xformers backend."
        super().build_model()

    def _calc_mean_maskrate_to_u_LUT(self, T: int):

        L = self.cfg.masking.span_len

        u2mean = [0.0]          v = (T - L) / float(T)
        for u in range(1, T):
            u2mean.append(1 - v)
            v *= (T - L - u) / (T - u)  
        mean2u = []
        for maskperc in range(101):
            maskrate = maskperc / float(100)
            u = int(np.searchsorted(u2mean, maskrate))
            mean2u.append(u)

        return mean2u

    def _non_spans_mask(self, mask_probs: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        num_token_masked = (T * mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((B, T), device=device).argsort(dim=-1)
        return batch_randperm < rearrange(num_token_masked, 'b -> b 1')

    def _spans_mask(self, mask_probs: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        rounded_probs = torch.round(100 * mask_probs).long()
        k = self.mean_maskrate_to_u[rounded_probs].clamp(min=1)  
                batch_randperm = torch.rand((B, T), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(k, 'b -> b 1')
        B, T = mask.shape
        shifted_mask = mask.clone()
        for _ in range(self.cfg.masking.span_len - 1):
            shifted_mask = torch.concat((torch.full((B, 1), False, device=device), shifted_mask[:, :-1]), dim=1)
            mask = torch.logical_or(mask, shifted_mask)

        return mask

    def _get_mask(self, mask_probs: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        if self.cfg.masking.span_len <= 1:
            return self._non_spans_mask(mask_probs, B, T, device)

        return self._spans_mask(mask_probs, B, T, device)

    def _compute_cross_entropy_magnet(self, logits: torch.Tensor,
                                      targets: torch.Tensor, mask: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        logits_k = logits[:, stage, ...].contiguous().view(-1, logits.size(-1))          targets_k = targets[:, stage, ...].contiguous().view(-1)          mask_k = mask[:, stage, ...].contiguous().view(-1)  
        IGNORE_IDX = -1
        targets_k[~mask_k] = IGNORE_IDX
        q_ce = F.cross_entropy(logits_k, targets_k, ignore_index=IGNORE_IDX)

        ce += q_ce
        return ce

    def run_step(self, idx: int, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]], metrics: dict) -> dict:

    More information can be found in the MAGNeT model card.
