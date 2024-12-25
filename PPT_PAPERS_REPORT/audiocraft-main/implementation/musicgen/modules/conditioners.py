
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
import logging
import math
from pathlib import Path
import random
import re
import typing as tp
import warnings

import einops
import flashy
from num2words import num2words
import spacy
from transformers import RobertaTokenizer, T5EncoderModel, T5Tokenizer  import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .chroma import ChromaExtractor
from .streaming import StreamingModule
from .transformer import create_sin_embedding, StreamingTransformer
from ..data.audio import audio_read
from ..data.audio_dataset import SegmentInfo
from ..data.audio_utils import convert_audio
from ..environment import AudioCraftEnvironment
from ..quantization import ResidualVectorQuantizer
from ..utils.autocast import TorchAutocast
from ..utils.cache import EmbeddingCache
from ..utils.utils import collate, hash_trick, length_to_mask, load_clap_state_dict, warn_once


logger = logging.getLogger(__name__)
TextCondition = tp.Optional[str]  ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  

class WavCondition(tp.NamedTuple):
    wav: torch.Tensor
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


class JointEmbedCondition(tp.NamedTuple):
    wav: torch.Tensor
    text: tp.List[tp.Optional[str]]
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


@dataclass
class ConditioningAttributes:
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    wav: tp.Dict[str, WavCondition] = field(default_factory=dict)
    joint_embed: tp.Dict[str, JointEmbedCondition] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def joint_embed_attributes(self):
        return self.joint_embed.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "wav": self.wav_attributes,
            "joint_embed": self.joint_embed_attributes,
        }

    def to_flat_dict(self):
        return {
            **{f"text.{k}": v for k, v in self.text.items()},
            **{f"wav.{k}": v for k, v in self.wav.items()},
            **{f"joint_embed.{k}": v for k, v in self.joint_embed.items()}
        }

    @classmethod
    def from_flat_dict(cls, x):
        out = cls()
        for k, v in x.items():
            kind, att = k.split(".")
            out[kind][att] = v
        return out


class SegmentWithAttributes(SegmentInfo):
    def to_condition_attributes(self) -> ConditioningAttributes:
        raise NotImplementedError()


def nullify_condition(condition: ConditionType, dim: int = 1):
    assert dim != 0, "dim cannot be the batch dimension!"
    assert isinstance(condition, tuple) and \
        isinstance(condition[0], torch.Tensor) and \
        isinstance(condition[1], torch.Tensor), "'nullify_condition' got an unexpected input type!"
    cond, mask = condition
    B = cond.shape[0]
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0. * out[..., :1]
    out = out.transpose(dim, last_dim)
    mask = torch.zeros((B, 1), device=out.device).int()
    assert cond.dim() == out.dim()
    return out, mask


def nullify_wav(cond: WavCondition) -> WavCondition:
    null_wav, _ = nullify_condition((cond.wav, torch.zeros_like(cond.wav)), dim=cond.wav.dim() - 1)
    return WavCondition(
        wav=null_wav,
        length=torch.tensor([0] * cond.wav.shape[0], device=cond.wav.device),
        sample_rate=cond.sample_rate,
        path=[None] * cond.wav.shape[0],
        seek_time=[None] * cond.wav.shape[0],
    )


def nullify_joint_embed(embed: JointEmbedCondition) -> JointEmbedCondition:
    null_wav, _ = nullify_condition((embed.wav, torch.zeros_like(embed.wav)), dim=embed.wav.dim() - 1)
    return JointEmbedCondition(
        wav=null_wav, text=[None] * len(embed.text),
        length=torch.LongTensor([0]).to(embed.wav.device),
        sample_rate=embed.sample_rate,
        path=[None] * embed.wav.shape[0],
        seek_time=[0] * embed.wav.shape[0],
    )


def _drop_description_condition(conditions: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        for condition in conditions:
        assert 'description' in condition.text.keys()
        assert 'self_wav' in condition.wav.keys()
    return AttributeDropout(p={'text': {'description': 1.0},
                               'wav': {'self_wav': 0.0}})(conditions)


class Tokenizer:
    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class WhiteSpaceTokenizer(Tokenizer):
    PUNCTUATION = "?:!.,;"

    def __init__(self, n_bins: int, pad_idx: int = 0, language: str = "en_core_web_sm",
                 lemma: bool = True, stopwords: bool = True) -> None:
        self.n_bins = n_bins
        self.pad_idx = pad_idx
        self.lemma = lemma
        self.stopwords = stopwords
        try:
            self.nlp = spacy.load(language)
        except IOError:
            spacy.cli.download(language)              self.nlp = spacy.load(language)

    @tp.no_type_check
    def __call__(self, texts: tp.List[tp.Optional[str]],
                 return_text: bool = False) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        output, lengths = [], []
        texts = deepcopy(texts)
        for i, text in enumerate(texts):
                        if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

                        text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)                          text = self.nlp(text)                          if self.stopwords:
                text = [w for w in text if not w.is_stop]                          text = [w for w in text if w.text not in self.PUNCTUATION]                          text = [getattr(t, "lemma_" if self.lemma else "text") for t in text]  
            texts[i] = " ".join(text)
            lengths.append(len(text))
                        tokens = torch.Tensor([hash_trick(w, self.n_bins) for w in text])
            output.append(tokens)

        mask = length_to_mask(torch.IntTensor(lengths)).int()
        padded_output = pad_sequence(output, padding_value=self.pad_idx).int().t()
        if return_text:
            return padded_output, mask, texts          return padded_output, mask


class NoopTokenizer(Tokenizer):
    def __init__(self, n_bins: int, pad_idx: int = 0):
        self.n_bins = n_bins
        self.pad_idx = pad_idx

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        output, lengths = [], []
        for text in texts:
                        if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                output.append(hash_trick(text, self.n_bins))
                lengths.append(1)

        tokens = torch.LongTensor(output).unsqueeze(1)
        mask = length_to_mask(torch.IntTensor(lengths)).int()
        return tokens, mask


class BaseConditioner(nn.Module):
    def __init__(self, dim: int, output_dim: int):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs) -> tp.Any:
        raise NotImplementedError()

    def forward(self, inputs: tp.Any) -> ConditionType:
        raise NotImplementedError()


class TextConditioner(BaseConditioner):
    ...


class LUTConditioner(TextConditioner):
    def __init__(self, n_bins: int, dim: int, output_dim: int, tokenizer: str, pad_idx: int = 0):
        super().__init__(dim, output_dim)
        self.embed = nn.Embedding(n_bins, dim)
        self.tokenizer: Tokenizer
        if tokenizer == 'whitespace':
            self.tokenizer = WhiteSpaceTokenizer(n_bins, pad_idx=pad_idx)
        elif tokenizer == 'noop':
            self.tokenizer = NoopTokenizer(n_bins, pad_idx=pad_idx)
        else:
            raise ValueError(f"unrecognized tokenizer `{tokenizer}`.")

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        device = self.embed.weight.device
        tokens, mask = self.tokenizer(x)
        tokens, mask = tokens.to(device), mask.to(device)
        return tokens, mask

    def forward(self, inputs: tp.Tuple[torch.Tensor, torch.Tensor]) -> ConditionType:
        tokens, mask = inputs
        embeds = self.embed(tokens)
        embeds = self.output_proj(embeds)
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask


class T5Conditioner(TextConditioner):
    MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(self, name: str, output_dim: int, finetune: bool, device: str,
                 autocast_dtype: tp.Optional[str] = 'float32', word_dropout: float = 0.,
                 normalize_text: bool = False):
        assert name in self.MODELS, f"Unrecognized t5 model name (should in {self.MODELS})"
        super().__init__(self.MODELS_DIMS[name], output_dim)
        self.device = device
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout
        if autocast_dtype is None or self.device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            if self.device != 'cpu':
                logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)
                        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(name)
                t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(previous_level)
        if finetune:
            self.t5 = t5
        else:
                                    self.__dict__['t5'] = t5.to(device)

        self.normalize_text = normalize_text
        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
                entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True)
        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(entries, return_tensors='pt', padding=True).to(self.device)
        mask = inputs['attention_mask']
        mask[empty_idx, :] = 0          return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs['attention_mask']
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask


class WaveformConditioner(BaseConditioner):
    def __init__(self, dim: int, output_dim: int, device: tp.Union[torch.device, str]):
        super().__init__(dim, output_dim)
        self.device = device
                self._use_masking = True

    def tokenize(self, x: WavCondition) -> WavCondition:
        wav, length, sample_rate, path, seek_time = x
        assert length is not None
        return WavCondition(wav.to(self.device), length.to(self.device), sample_rate, path, seek_time)

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, x: WavCondition) -> ConditionType:
        wav, lengths, *_ = x
        with torch.no_grad():
            embeds = self._get_wav_embedding(x)
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)

        if lengths is not None and self._use_masking:
            lengths = lengths / self._downsampling_factor()
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()          else:
            mask = torch.ones_like(embeds[..., 0])
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask


class ChromaStemConditioner(WaveformConditioner):
    def __init__(self, output_dim: int, sample_rate: int, n_chroma: int, radix2_exp: int,
                 duration: float, match_len_on_eval: bool = True, eval_wavs: tp.Optional[str] = None,
                 n_eval_wavs: int = 0, cache_path: tp.Optional[tp.Union[str, Path]] = None,
                 device: tp.Union[torch.device, str] = 'cpu', **kwargs):
        from demucs import pretrained
        super().__init__(dim=n_chroma, output_dim=output_dim, device=device)
        self.autocast = TorchAutocast(enabled=device != 'cpu', device_type=self.device, dtype=torch.float32)
        self.sample_rate = sample_rate
        self.match_len_on_eval = match_len_on_eval
        if match_len_on_eval:
            self._use_masking = False
        self.duration = duration
        self.__dict__['demucs'] = pretrained.get_model('htdemucs').to(device)
        stem_sources: list = self.demucs.sources          self.stem_indices = torch.LongTensor([stem_sources.index('vocals'), stem_sources.index('other')]).to(device)
        self.chroma = ChromaExtractor(sample_rate=sample_rate, n_chroma=n_chroma,
                                      radix2_exp=radix2_exp, **kwargs).to(device)
        self.chroma_len = self._get_chroma_len()
        self.eval_wavs: tp.Optional[torch.Tensor] = self._load_eval_wavs(eval_wavs, n_eval_wavs)
        self.cache = None
        if cache_path is not None:
            self.cache = EmbeddingCache(Path(cache_path) / 'wav', self.device,
                                        compute_embed_fn=self._get_full_chroma_for_cache,
                                        extract_embed_fn=self._extract_chroma_chunk)

    def _downsampling_factor(self) -> int:
        return self.chroma.winhop

    def _load_eval_wavs(self, path: tp.Optional[str], num_samples: int) -> tp.Optional[torch.Tensor]:
        if path is None:
            return None

        logger.info(f"Loading evaluation wavs from {path}")
        from audiocraft.data.audio_dataset import AudioDataset
        dataset: AudioDataset = AudioDataset.from_meta(
            path, segment_duration=self.duration, min_audio_duration=self.duration,
            sample_rate=self.sample_rate, channels=1)

        if len(dataset) > 0:
            eval_wavs = dataset.collater([dataset[i] for i in range(num_samples)]).to(self.device)
            logger.info(f"Using {len(eval_wavs)} evaluation wavs for chroma-stem conditioner")
            return eval_wavs
        else:
            raise ValueError("Could not find evaluation wavs, check lengths of wavs")

    def reset_eval_wavs(self, eval_wavs: tp.Optional[torch.Tensor]) -> None:
        self.eval_wavs = eval_wavs

    def has_eval_wavs(self) -> bool:
        return self.eval_wavs is not None

    def _sample_eval_wavs(self, num_samples: int) -> torch.Tensor:
        dummy_wav = torch.zeros((1, int(self.sample_rate * self.duration)), device=self.device)
        dummy_chr = self.chroma(dummy_wav)
        return dummy_chr.shape[1]

    @torch.no_grad()
    def _get_stemmed_wav(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        with self.autocast:
            return self.chroma(wav)

    @torch.no_grad()
    def _compute_wav_embedding(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav, sr = audio_read(path)
        wav = wav[None].to(self.device)
        wav = convert_audio(wav, sr, self.sample_rate, to_channels=1)
        chroma = self._compute_wav_embedding(wav, self.sample_rate)[0]
        return chroma

    def _extract_chroma_chunk(self, full_chroma: torch.Tensor, x: WavCondition, idx: int) -> torch.Tensor:
        The conditioner will either extract the embedding on-the-fly computing it from the condition wav directly
        or will rely on the embedding cache to load the pre-computed embedding if relevant.
        x = super().tokenize(x)
        no_undefined_paths = all(p is not None for p in x.path)
        if self.cache is not None and no_undefined_paths:
            paths = [Path(p) for p in x.path if p is not None]
            self.cache.populate_embed_cache(paths, x)
        return x


class FeatureExtractor(WaveformConditioner):
    def __init__(
        self, model_name: str,
        sample_rate: int, encodec_checkpoint: str, encodec_n_q: int, length: float,
        dim: int, output_dim: int, device: tp.Union[torch.device, str],
        compute_mask: bool = True,
        use_middle_of_segment: bool = False, ds_rate_compression: int = 640,
        num_codebooks_lm: int = 4
    ):
        assert model_name in ['encodec', 'mert']
        if model_name == 'encodec':
            from ..solvers.compression import CompressionSolver
            feat_extractor = CompressionSolver.model_from_checkpoint(encodec_checkpoint, device)
        elif model_name == 'mert':
            from transformers import AutoModel
            feat_extractor = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        super().__init__(
            dim=dim,
            output_dim=output_dim,
            device=device
        )
        self.sample_rate = sample_rate
        self.compute_mask = compute_mask
        self.feat_extractor: nn.Module
        self.embed: tp.Union[nn.ModuleList, nn.Linear]
        if model_name == 'encodec':
            self.__dict__["feat_extractor"] = feat_extractor.to(device)
            self.encodec_n_q = encodec_n_q
            self.embed = nn.ModuleList([nn.Embedding(feat_extractor.cardinality, dim) for _ in range(encodec_n_q)])
        if model_name == 'mert':
            self.__dict__["feat_extractor"] = feat_extractor.eval().to(device)
            self.embed = nn.Linear(768, dim)          self.length_subwav = int(length * sample_rate)
        self.ds_rate_compression = ds_rate_compression
        self.model_name = model_name
        self.use_middle_of_segment = use_middle_of_segment
        self.num_codebooks_lm = num_codebooks_lm

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        if x.wav.shape[-1] == 1:
            self.temp_mask = None
            return torch.zeros(x.wav.shape[0], 1, self.dim, device=self.device)
        else:
            with torch.no_grad():
                if self.use_middle_of_segment:
                    start = int((x.wav.shape[-1] - self.length_subwav) / 2)
                    wav = x.wav[:, :, start:start+self.length_subwav]
                else:
                    start = random.randint(0, x.wav.shape[-1] - self.length_subwav)
                    wav = x.wav[:, :, start:start+self.length_subwav]
                if self.compute_mask:
                    self.temp_mask = self._get_mask_wav(x, start)
                if self.model_name == 'encodec':
                    tokens = self.feat_extractor.encode(wav)[0]                  elif self.model_name == 'mert':
                    wav = convert_audio(wav, from_rate=x.sample_rate[0], to_rate=24000, to_channels=1)
                    embeds = self.feat_extractor(wav.squeeze(-2)).last_hidden_state
            if self.model_name == 'encodec':
                tokens = tokens[:, :self.encodec_n_q]
                embeds = sum([self.embed[k](tokens[:, k]) for k in range(self.encodec_n_q)])              else:
                embeds = self.embed(embeds)

            return embeds  
    def _downsampling_factor(self):
        if self.model_name == 'encodec':
            return self.sample_rate / self.feat_extractor.frame_rate
        elif self.model_name == 'mert':
            return self.sample_rate / 75

    def _get_mask_wav(self, x: WavCondition, start: int) -> tp.Union[torch.Tensor, None]:
        if x.wav.shape[-1] == 1:
            return None
        total_length = int(x.wav.shape[-1] / self.ds_rate_compression)
        mask_length = int(self.length_subwav / self.ds_rate_compression)
        start = int(start / self.ds_rate_compression)
        mask = torch.ones(x.wav.shape[0], self.num_codebooks_lm,
                          total_length, device=self.device, dtype=torch.bool)
        mask[:, :, start:start+mask_length] = 0
        return mask


class StyleConditioner(FeatureExtractor):
    def __init__(self, transformer_scale: str = 'default', ds_factor: int = 15, encodec_n_q: int = 4,
                 n_q_out: int = 6, eval_q: int = 3, q_dropout: bool = True, bins: int = 1024,
                 varying_lengths: tp.List[float] = [1.5, 4.5],
                 batch_norm: bool = True, rvq_threshold_ema_dead_code: float = 0.1,
                 **kwargs):
        tr_args: tp.Dict[str, tp.Any]
        if transformer_scale == 'xsmall':
            tr_args = {'d_model': 256, 'num_heads': 8, 'num_layers': 4}
        elif transformer_scale == 'large':
            tr_args = {'d_model': 1024, 'num_heads': 16, 'num_layers': 24}
        elif transformer_scale == 'default':
            tr_args = {'d_model': 512, 'num_heads': 8, 'num_layers': 8}
        elif transformer_scale == 'none':
            tr_args = {'d_model': 512}
        tr_args.update({
            'memory_efficient': True, 'activation': 'gelu',
            'norm_first': True, 'causal': False, 'layer_scale': None,
            'bias_ff': False, 'bias_attn': False,
        })
        dim = tr_args['d_model']
        super().__init__(dim=dim, encodec_n_q=encodec_n_q, **kwargs)

        self.ds_factor = ds_factor
        if transformer_scale == 'none':
            self.transformer = None
        else:
            self.transformer = StreamingTransformer(dim_feedforward=int(4 * dim), **tr_args)
        self.n_q_out = n_q_out
        self.eval_q = eval_q
        self.rvq = None
        if n_q_out > 0:
            self.rvq = ResidualVectorQuantizer(dim, n_q=n_q_out, q_dropout=q_dropout, bins=bins,
                                               threshold_ema_dead_code=rvq_threshold_ema_dead_code)
        self.autocast = TorchAutocast(enabled=self.device != 'cpu', device_type=self.device, dtype=torch.float32)
        self.varying_lengths = varying_lengths
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim, affine=False)
        self.mask = None

    def _get_wav_embedding(self, wav: WavCondition) -> torch.Tensor:
        with self.autocast:
                        if self.varying_lengths and self.training:
                assert len(self.varying_lengths) == 2
                length = random.uniform(self.varying_lengths[0], self.varying_lengths[1])
                self.length_subwav = int(length * self.sample_rate)
            z1 = super()._get_wav_embedding(wav)
            if self.compute_mask:
                self.mask = self.temp_mask              self.temp_mask = None

            if self.transformer is not None:
                out1 = self.transformer(z1)
            else:
                out1 = z1
            if self.batch_norm:
                out1 = self.batch_norm(out1.transpose(1, 2)).transpose(1, 2)
                        if self.rvq:
                if self.training:
                    self.rvq.set_num_codebooks(self.n_q_out)
                else:
                    self.rvq.set_num_codebooks(self.eval_q)
                out1 = self.rvq(out1.transpose(1, 2), frame_rate=1.)
                if self.training:
                    flashy.distrib.average_tensors(self.rvq.buffers())
                out1 = out1.x.transpose(1, 2)
                        out1 = out1[:, ::self.ds_factor]

        return out1

    def set_params(self, eval_q: int = 3,
                   excerpt_length: float = 3.0,
                   ds_factor: tp.Optional[int] = None, encodec_n_q: tp.Optional[int] = None):
        self.eval_q = eval_q
        self.length_subwav = int(excerpt_length * self.sample_rate)
        if ds_factor is not None:
            self.ds_factor = ds_factor
        if encodec_n_q is not None:
            self.encodec_n_q = encodec_n_q

    def _downsampling_factor(self):
        df = super()._downsampling_factor()
        return df * self.ds_factor

    def forward(self, x: WavCondition) -> ConditionType:
        wav, lengths, *_ = x

        embeds = self._get_wav_embedding(x)
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)

        lengths = lengths / self._downsampling_factor()
        mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  
        embeds = (embeds * mask.unsqueeze(2).to(self.device))

        return embeds, mask


class JointEmbeddingConditioner(BaseConditioner):
    def __init__(self, dim: int, output_dim: int, device: str, attribute: str,
                 autocast_dtype: tp.Optional[str] = 'float32', quantize: bool = True,
                 n_q: int = 12, bins: int = 1024, **kwargs):
        super().__init__(dim=dim, output_dim=output_dim)
        self.device = device
        self.attribute = attribute
        if autocast_dtype is None or device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            logger.warning("JointEmbeddingConditioner has no autocast, this might lead to NaN.")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"JointEmbeddingConditioner will be evaluated with autocast as {autocast_dtype}.")
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)
                self.quantizer: tp.Optional[ResidualVectorQuantizer] = None
        if quantize:
            self.quantizer = ResidualVectorQuantizer(dim, n_q=n_q, bins=bins, **kwargs)

    def _get_embed(self, x: JointEmbedCondition) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def forward(self, x: JointEmbedCondition) -> ConditionType:
        with self.autocast:
            embed, empty_idx = self._get_embed(x)
            if self.quantizer is not None:
                embed = embed.view(-1, self.dim, 1)
                q_res = self.quantizer(embed, frame_rate=1)
                out_embed = q_res.x.view(-1, self.dim)
            else:
                out_embed = embed
            out_embed = self.output_proj(out_embed).view(-1, 1, self.output_dim)
            mask = torch.ones(*out_embed.shape[:2], device=out_embed.device)
            mask[empty_idx, :] = 0              out_embed = (out_embed * mask.unsqueeze(-1))
            return out_embed, mask

    def tokenize(self, x: JointEmbedCondition) -> JointEmbedCondition:
        return x


class CLAPEmbeddingConditioner(JointEmbeddingConditioner):
    def __init__(self, dim: int, output_dim: int, device: str, attribute: str,
                 quantize: bool, n_q: int, bins: int, checkpoint: tp.Union[str, Path], model_arch: str,
                 enable_fusion: bool, sample_rate: int, max_audio_length: int, audio_stride: int,
                 normalize: bool, text_p: bool, batch_size: tp.Optional[int] = None,
                 autocast_dtype: tp.Optional[str] = 'float32', cache_path: tp.Optional[str] = None, **kwargs):
        try:
            import laion_clap          except ImportError:
            raise ImportError("Please install CLAP to use the CLAPEmbeddingConditioner: 'pip install laion_clap'")
        warnings.warn("Sample rate for CLAP conditioner was fixed in version v1.1.0, (from 44.1 to 48 kHz). "
                      "Please retrain all models.")
        checkpoint = AudioCraftEnvironment.resolve_reference_path(checkpoint)
        clap_tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        clap_model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=model_arch)
        load_clap_state_dict(clap_model, checkpoint)
        clap_model.eval()
        clap_model.to(device)
        super().__init__(dim=dim, output_dim=output_dim, device=device, attribute=attribute,
                         autocast_dtype=autocast_dtype, quantize=quantize, n_q=n_q, bins=bins,
                         **kwargs)
        self.checkpoint = checkpoint
        self.enable_fusion = enable_fusion
        self.model_arch = model_arch
        self.clap: laion_clap.CLAP_Module
        self.clap_tokenize: RobertaTokenizer
        self.clap_sample_rate = sample_rate
        self.clap_max_frames = int(self.clap_sample_rate * max_audio_length)
        self.clap_stride = int(self.clap_sample_rate * audio_stride)
        self.batch_size = batch_size or 1
        self.normalize = normalize
        self.text_p = text_p
        self.__dict__['clap_tokenize'] = clap_tokenize
        self.__dict__['clap'] = clap_model
        self.wav_cache, self.text_cache = None, None
        if cache_path is not None:
            self.wav_cache = EmbeddingCache(Path(cache_path) / 'wav', self.device,
                                            compute_embed_fn=self._get_wav_embedding_for_cache,
                                            extract_embed_fn=self._extract_wav_embedding_chunk)
            self.text_cache = EmbeddingCache(Path(cache_path) / 'text', self.device,
                                             compute_embed_fn=self._get_text_embedding_for_cache)

    def _tokenizer(self, texts: tp.Union[str, tp.List[str]]) -> dict:
                return self.clap_tokenize(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")

    def _compute_text_embedding(self, text: tp.List[str]) -> torch.Tensor:
        with torch.no_grad():
            embed = self.clap.get_text_embedding(text, tokenizer=self._tokenizer, use_tensor=True)
            return embed.view(embed.size(0), 1, embed.size(-1))

    def _get_text_embedding_for_cache(self, path: tp.Union[Path, str],
                                      x: JointEmbedCondition, idx: int) -> torch.Tensor:

        Args:
            wav (torch.Tensor): Audio wav, of shape [B, C, T].
            length (torch.Tensor): Actual length of the audio for each item in the batch, of shape [B].
            sample_rates (list[int]): Sample rates for each sample in the batch
        Returns:
            torch.Tensor: Audio wav of shape [B, T].

        Since CLAP operates on a fixed sequence length audio inputs and we need to process longer audio sequences,
        we calculate the wav embeddings on `clap_max_frames` windows with `clap_stride`-second stride and
        average the resulting embeddings.

        Args:
            wav (torch.Tensor): Audio wav, of shape [B, C, T].
            length (torch.Tensor): Actual length of the audio for each item in the batch, of shape [B].
            sample_rates (list[int]): Sample rates for each sample in the batch.
            reduce_mean (bool): Whether to get the average tensor.
        Returns:
            torch.Tensor: Audio embedding of shape [B, F, D], F being the number of chunks, D the dimension.
        The embedding is computed on a given audio read from file.

        Args:
            path (str or Path): Path to the full audio file.
        Returns:
            torch.Tensor: Single-item tensor of shape [F, D], F being the number of chunks, D the dimension.

        Args:
            full_embed (torch.Tensor): CLAP embedding computed on the full wave, of shape [F, D].
            x (JointEmbedCondition): Joint embedding condition for the full batch.
            idx (int): Index considered for the given embedding to extract.
        Returns:
            torch.Tensor: Wav embedding averaged on sliding window, of shape [1, D].
        no_nullified_cond = x.wav.shape[-1] > 1          if self.text_cache is not None and no_nullified_cond:
            assert all(p is not None for p in x.path), "Cache requires all JointEmbedCondition paths to be provided"
            paths = [Path(p) for p in x.path if p is not None]
            embed = self.text_cache.get_embed_from_cache(paths, x)
        else:
            text = [xi if xi is not None else "" for xi in x.text]
            embed = self._compute_text_embedding(text)
        if self.normalize:
            embed = torch.nn.functional.normalize(embed, p=2.0, dim=-1)
        return embed

    def _get_wav_embedding(self, x: JointEmbedCondition) -> torch.Tensor:
                use_text_embed = random.random() < self.text_p
        if self.training and not use_text_embed:
            embed = self._get_wav_embedding(x)
            empty_idx = torch.LongTensor([])          else:
            embed = self._get_text_embedding(x)
            empty_idx = torch.LongTensor([i for i, xi in enumerate(x.text) if xi is None or xi == ""])
        return embed, empty_idx


def dropout_condition(sample: ConditioningAttributes, condition_type: str, condition: str) -> ConditioningAttributes:
    if condition_type not in ['text', 'wav', 'joint_embed']:
        raise ValueError(
            "dropout_condition got an unexpected condition type!"
            f" expected 'text', 'wav' or 'joint_embed' but got '{condition_type}'"
        )

    if condition not in getattr(sample, condition_type):
        raise ValueError(
            "dropout_condition received an unexpected condition!"
            f" expected wav={sample.wav.keys()} and text={sample.text.keys()}"
            f" but got '{condition}' of type '{condition_type}'!"
        )

    if condition_type == 'wav':
        wav_cond = sample.wav[condition]
        sample.wav[condition] = nullify_wav(wav_cond)
    elif condition_type == 'joint_embed':
        embed = sample.joint_embed[condition]
        sample.joint_embed[condition] = nullify_joint_embed(embed)
    else:
        sample.text[condition] = None

    return sample


class DropoutModule(nn.Module):
    This is different from the behavior of ClassifierFreeGuidanceDropout as this allows for attributes
    to be dropped out separately. For example, "artist" can be dropped while "genre" remains.
    This is in contrast to ClassifierFreeGuidanceDropout where if "artist" is dropped "genre"
    must also be dropped.

    Args:
        p (tp.Dict[str, float]): A dict mapping between attributes and dropout probability. For example:
            ...
            "genre": 0.1,
            "artist": 0.5,
            "wav": 0.25,
            ...
        active_on_eval (bool, optional): Whether the dropout is active at eval. Default to False.
        seed (int, optional): Random seed.
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after certain attributes were set to None.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after all attributes were set to None.

    Args:
        conditioners (dict): Dictionary of conditioners.
        device (torch.device or str, optional): Device for conditioners and output condition types.
        This should be called before starting any real GPU work to avoid synchronization points.
        This will return a dict matching conditioner names to their arbitrary tokenized representations.

        Args:
            inputs (list[ConditioningAttributes]): List of ConditioningAttributes objects containing
                text and wav conditions.
        The output is for example:
        {
            "genre": (torch.Tensor([B, 1, D_genre]), torch.Tensor([B, 1])),
            "description": (torch.Tensor([B, T_desc, D_desc]), torch.Tensor([B, T_desc])),
            ...
        }

        Args:
            tokenized (dict): Dict of tokenized representations as returned by `tokenize()`.
        are the attributes and the values are the aggregated input per attribute.
        For example:
        Input:
        [
            ConditioningAttributes(text={"genre": "Rock", "description": "A rock song with a guitar solo"}, wav=...),
            ConditioningAttributes(text={"genre": "Hip-hop", "description": "A hip-hop verse"}, wav=...),
        ]
        Output:
        {
            "genre": ["Rock", "Hip-hop"],
            "description": ["A rock song with a guitar solo", "A hip-hop verse"]
        }

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, list[str, optional]]: A dictionary mapping an attribute name to text batch.
        and the values are Tensors of wavs according to said attributes.

        *Note*: by the time the samples reach this function, each sample should have some waveform
        inside the "wav" attribute. It should be either:
        1. A real waveform
        2. A null waveform due to the sample having no similar waveforms (nullified by the dataset)
        3. A null waveform due to it being dropped in a dropout module (nullified by dropout)

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        and the values are Tensors of pre-computed embeddings and the corresponding text attributes.

        Args:
            samples (list[ConditioningAttributes]): List of ConditioningAttributes samples.
        Returns:
            A dictionary mapping an attribute name to joint embeddings.
    to the actual model input.

    Args:
        fuse2cond (tp.Dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["description"],
                "sum": ["genre", "bpm"],
                "cross": ["description"],
            }
        cross_attention_pos_emb (bool, optional): Use positional embeddings in cross attention.
        cross_attention_pos_emb_scale (int): Scale for positional embeddings in cross attention if used.

        Args:
            input (torch.Tensor): Transformer input.
            conditions (dict[str, ConditionType]): Dict of conditions.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The first tensor is the transformer input
                after the conditions have been fused. The second output tensor is the tensor
                used for cross-attention or None if no cross attention inputs exist.
