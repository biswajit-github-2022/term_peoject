
from abc import ABC, abstractmethod
import logging
import math
from pathlib import Path
import typing as tp

from einops import rearrange
import numpy as np
import torch
from torch import nn
from transformers import EncodecModel as HFEncodecModel

from .. import quantization as qt


logger = logging.getLogger()


class CompressionModel(ABC, nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor):
        ...

    @staticmethod
    def get_pretrained(
            name: str, device: tp.Union[torch.device, str] = 'cpu'
            ) -> 'CompressionModel':

        from . import builders, loaders
        model: CompressionModel
        if name in ['dac_44khz', 'dac_24khz']:
            model_type = name.split('_')[1]
            logger.info("Getting pretrained compression model from DAC %s", model_type)
            model = DAC(model_type)
        elif name in ['debug_compression_model']:
            logger.info("Getting pretrained compression model for debug")
            model = builders.get_debug_compression_model()
        elif Path(name).exists():
                                    model = loaders.load_compression_model(name, device=device)
        else:
            logger.info("Getting pretrained compression model from HF %s", name)
            hf_model = HFEncodecModel.from_pretrained(name)
            model = HFEncodecCompressionModel(hf_model).to(device)
        return model.to(device).eval()


class EncodecModel(CompressionModel):
            frame_rate: float = 0
    sample_rate: int = 0
    channels: int = 0

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 quantizer: qt.BaseQuantizer,
                 frame_rate: int,
                 sample_rate: int,
                 channels: int,
                 causal: bool = False,
                 renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        if self.causal:
                                    assert not self.renormalize, 'Causal model does not support renormalize'

    @property
    def total_codebooks(self):
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        return self.quantizer.bins

    def preprocess(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(self,
                    x: torch.Tensor,
                    scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)

        emb = self.encoder(x)
        q_res = self.quantizer(emb, self.frame_rate)
        out = self.decoder(q_res.x)

                assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = self.postprocess(out, scale)

        return q_res

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb)
        return codes, scale

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
                return out

    def decode_latent(self, codes: torch.Tensor):
        return self.model.quantizer.from_codes(codes)[0]

    @property
    def channels(self) -> int:
        return 1

    @property
    def frame_rate(self) -> float:
        return self.model.sample_rate / self.model.hop_length

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def cardinality(self) -> int:
        return self.model.codebook_size

    @property
    def num_codebooks(self) -> int:
        return self.n_quantizers

    @property
    def total_codebooks(self) -> int:
        return self.model.n_codebooks

    def set_num_codebooks(self, n: int):
        assert n >= 1
        assert n <= self.total_codebooks
        self.n_quantizers = n


class HFEncodecCompressionModel(CompressionModel):
    def __init__(self, model: HFEncodecModel):
        super().__init__()
        self.model = model
        bws = self.model.config.target_bandwidths
        num_codebooks = [
            bw * 1000 / (self.frame_rate * math.log2(self.cardinality))
            for bw in bws
        ]
        deltas = [nc - int(nc) for nc in num_codebooks]
                assert all(deltas) <= 1e-3, deltas
        self.possible_num_codebooks = [int(nc) for nc in num_codebooks]
        self.set_num_codebooks(max(self.possible_num_codebooks))

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
                raise NotImplementedError("Forward and training with HF EncodecModel not supported.")

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        bandwidth_index = self.possible_num_codebooks.index(self.num_codebooks)
        bandwidth = self.model.config.target_bandwidths[bandwidth_index]
        res = self.model.encode(x, None, bandwidth)
        assert len(res[0]) == 1
        assert len(res[1]) == 1
        return res[0][0], res[1][0]

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        if scale is None:
            scales = [None]          else:
            scales = scale          res = self.model.decode(codes[None], scales)
        return res[0]

    def decode_latent(self, codes: torch.Tensor):
    will be applied independently to the left and right channels, and both codebooks
    will be interleaved. If the wrapped model returns a representation `[B, K ,T]` per
    channel, then the output will be `[B, K * 2, T]`  or `[B, K, T * 2]` depending on
    `per_timestep`.

    Args:
        model (CompressionModel): Compression model to wrap.
        per_timestep (bool): Whether to interleave on the timestep dimension
            or on the codebooks dimension.

        ..Warning:: this reports the number of codebooks after the interleaving
        of the codebooks!

        ..Warning:: this sets the number of codebooks before the interleaving!
        will be split into that many steps.
        raise NotImplementedError("Not supported by interleaved stereo wrapped models.")
