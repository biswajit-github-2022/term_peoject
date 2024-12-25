
import contextlib
from functools import partial
import logging
import os
import typing as tp

import torch
import torchmetrics

from ..data.audio_utils import convert_audio


logger = logging.getLogger(__name__)


class _patch_passt_stft:
    from generated samples and target samples.

    Args:
        pred_probs (torch.Tensor): Probabilities for each label obtained
            from a classifier on generated audio. Expected shape is [B, num_classes].
        target_probs (torch.Tensor): Probabilities for each label obtained
            from a classifier on target audio. Expected shape is [B, num_classes].
        epsilon (float): Epsilon value.
    Returns:
        kld (torch.Tensor): KLD loss between each generated sample and target pair.

    The KL divergence is measured between probability distributions
    of class predictions returned by a pre-trained audio classification model.
    When the KL-divergence is low, the generated audio is expected to
    have similar acoustic characteristics as the reference audio,
    according to the classifier.

        Args:
            x (torch.Tensor): Input audio tensor of shape [B, C, T].
            sizes (torch.Tensor): Actual audio sample length, of shape [B].
            sample_rates (torch.Tensor): Actual audio sample rate, of shape [B].
        Returns:
            probs (torch.Tensor): Probabilities over labels, of shape [B, num_classes].
        preds (generated) and target (ground-truth)
        Args:
            preds (torch.Tensor): Audio samples to evaluate, of shape [B, C, T].
            targets (torch.Tensor): Target samples to compare against, of shape [B, C, T].
            sizes (torch.Tensor): Actual audio sample length, of shape [B].
            sample_rates (torch.Tensor): Actual audio sample rate, of shape [B].
        weight: float = float(self.weight.item())          assert weight > 0, "Unable to compute with total number of comparisons <= 0"
        logger.info(f"Computing KL divergence on a total of {weight} samples")
        kld_pq = self.kld_pq_sum.item() / weight          kld_qp = self.kld_qp_sum.item() / weight          kld_both = kld_pq + kld_qp
        return {'kld': kld_pq, 'kld_pq': kld_pq, 'kld_qp': kld_qp, 'kld_both': kld_both}


class PasstKLDivergenceMetric(KLDivergenceMetric):
    def __init__(self, pretrained_length: tp.Optional[float] = None):
        super().__init__()
        self._initialize_model(pretrained_length)

    def _initialize_model(self, pretrained_length: tp.Optional[float] = None):
        try:
            if pretrained_length == 30:
                from hear21passt.base30sec import get_basic_model                  max_duration = 30
            elif pretrained_length == 20:
                from hear21passt.base20sec import get_basic_model                  max_duration = 20
            else:
                from hear21passt.base import get_basic_model                                  max_duration = 10
            min_duration = 0.15
            min_duration = 0.15
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install hear21passt to compute KL divergence: ",
                "pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19            )
        model_sample_rate = 32_000
        max_input_frames = int(max_duration * model_sample_rate)
        min_input_frames = int(min_duration * model_sample_rate)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            model = get_basic_model(mode='logits')
        return model, model_sample_rate, max_input_frames, min_input_frames

    def _process_audio(self, wav: torch.Tensor, sample_rate: int, wav_len: int) -> tp.List[torch.Tensor]:
        assert wav.dim() == 3, f"Unexpected number of dims for preprocessed wav: {wav.shape}"
        wav = wav.mean(dim=1)
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            with torch.no_grad(), _patch_passt_stft():
                logits = self.model(wav.to(self.device))
                probs = torch.softmax(logits, dim=-1)
                return probs

    def _get_label_distribution(self, x: torch.Tensor, sizes: torch.Tensor,
                                sample_rates: torch.Tensor) -> tp.Optional[torch.Tensor]:
        all_probs: tp.List[torch.Tensor] = []
        for i, wav in enumerate(x):
            sample_rate = int(sample_rates[i].item())
            wav_len = int(sizes[i].item())
            wav_segments = self._process_audio(wav, sample_rate, wav_len)
            for segment in wav_segments:
                probs = self._get_model_preds(segment).mean(dim=0)
                all_probs.append(probs)
        if len(all_probs) > 0:
            return torch.stack(all_probs, dim=0)
        else:
            return None
