
from typing import Literal

import torch
import torch.nn as nn


class WMDetectionLoss(nn.Module):
        Compute the masked sample-level detection loss
        (https://arxiv.org/pdf/2401.17264)

        Args:
            temperature: temperature for loss computation
            loss_type: bce or mse between outputs and original message
        Compute decoding loss
        Args:
            positive: outputs on watermarked samples [bsz, 2+nbits, time_steps]
            negative: outputs on not watermarked samples [bsz, 2+nbits, time_steps]
            mask: watermark mask [bsz, 1, time_steps]
            message: original message [bsz, nbits] or None
