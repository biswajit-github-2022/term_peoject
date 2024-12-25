
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
import random
import typing as tp

import numpy as np
import torch

from .info_audio_dataset import (
    InfoAudioDataset,
    get_keyword_or_keyword_list
)
from ..modules.conditioners import (
    ConditioningAttributes,
    SegmentWithAttributes,
    WavCondition,
)


EPS = torch.finfo(torch.float32).eps
TARGET_LEVEL_LOWER = -35
TARGET_LEVEL_UPPER = -15


@dataclass
class SoundInfo(SegmentWithAttributes):
    description: tp.Optional[str] = None
    self_wav: tp.Optional[torch.Tensor] = None

    @property
    def has_sound_meta(self) -> bool:
        return self.description is not None

    def to_condition_attributes(self) -> ConditioningAttributes:
        out = ConditioningAttributes()

        for _field in fields(self):
            key, value = _field.name, getattr(self, _field.name)
            if key == 'self_wav':
                out.wav[key] = value
            else:
                out.text[key] = value
        return out

    @staticmethod
    def attribute_getter(attribute):
        if attribute == 'description':
            preprocess_func = get_keyword_or_keyword_list
        else:
            preprocess_func = None
        return preprocess_func

    @classmethod
    def from_dict(cls, dictionary: dict, fields_required: bool = False):
        _dictionary: tp.Dict[str, tp.Any] = {}

                        post_init_attributes = ['self_wav']

        for _field in fields(cls):
            if _field.name in post_init_attributes:
                continue
            elif _field.name not in dictionary:
                if fields_required:
                    raise KeyError(f"Unexpected missing key: {_field.name}")
            else:
                preprocess_func: tp.Optional[tp.Callable] = cls.attribute_getter(_field.name)
                value = dictionary[_field.name]
                if preprocess_func:
                    value = preprocess_func(value)
                _dictionary[_field.name] = value
        return cls(**_dictionary)


class SoundDataset(InfoAudioDataset):
    def __init__(
        self,
        *args,
        info_fields_required: bool = True,
        external_metadata_source: tp.Optional[str] = None,
        aug_p: float = 0.,
        mix_p: float = 0.,
        mix_snr_low: int = -5,
        mix_snr_high: int = 5,
        mix_min_overlap: float = 0.5,
        **kwargs
    ):
        kwargs['return_info'] = True          super().__init__(*args, **kwargs)
        self.info_fields_required = info_fields_required
        self.external_metadata_source = external_metadata_source
        self.aug_p = aug_p
        self.mix_p = mix_p
        if self.aug_p > 0:
            assert self.mix_p > 0, "Expecting some mixing proportion mix_p if aug_p > 0"
            assert self.channels == 1, "SoundDataset with audio mixing considers only monophonic audio"
        self.mix_snr_low = mix_snr_low
        self.mix_snr_high = mix_snr_high
        self.mix_min_overlap = mix_min_overlap

    def _get_info_path(self, path: tp.Union[str, Path]) -> Path:
        info_path = Path(path).with_suffix('.json')
        if Path(info_path).exists():
            return info_path
        elif self.external_metadata_source and (Path(self.external_metadata_source) / info_path.name).exists():
            return Path(self.external_metadata_source) / info_path.name
        else:
            raise Exception(f"Unable to find a metadata JSON for path: {path}")

    def __getitem__(self, index):
        wav, info = super().__getitem__(index)
        info_data = info.to_dict()
        info_path = self._get_info_path(info.meta.path)
        if Path(info_path).exists():
            with open(info_path, 'r') as json_file:
                sound_data = json.load(json_file)
                sound_data.update(info_data)
                sound_info = SoundInfo.from_dict(sound_data, fields_required=self.info_fields_required)
                                if isinstance(sound_info.description, list):
                    sound_info.description = random.choice(sound_info.description)
        else:
            sound_info = SoundInfo.from_dict(info_data, fields_required=False)

        sound_info.self_wav = WavCondition(
            wav=wav[None], length=torch.tensor([info.n_frames]),
            sample_rate=[sound_info.sample_rate], path=[info.meta.path], seek_time=[info.seek_time])

        return wav, sound_info

    def collater(self, samples):
                wav, sound_info = super().collater(samples)          if self.aug_p > 0:
            wav, sound_info = mix_samples(wav, sound_info, self.aug_p, self.mix_p,
                                          snr_low=self.mix_snr_low, snr_high=self.mix_snr_high,
                                          min_overlap=self.mix_min_overlap)
        return wav, sound_info


def rms_f(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).mean(1).pow(0.5)


def normalize(audio: torch.Tensor, target_level: int = -25) -> torch.Tensor:

    Args:
        clean (torch.Tensor): Clean audio source to mix, of shape [B, T].
        noise (torch.Tensor): Noise audio source to mix, of shape [B, T].
        snr (int): SNR level when mixing.
        min_overlap (float): Minimum overlap between the two mixed sources.
        target_level (int): Gain level in dB.
        clipping_threshold (float): Threshold for clipping the audio.
    Returns:
        torch.Tensor: The mixed audio, of shape [B, T].
    if src_text == dst_text:
        return src_text
    return src_text + " " + dst_text


def mix_samples(wavs: torch.Tensor, infos: tp.List[SoundInfo], aug_p: float, mix_p: float,
                snr_low: int, snr_high: int, min_overlap: float):
        if mix_p == 0:
        return wavs, infos

    if random.uniform(0, 1) < aug_p:
                        assert wavs.size(1) == 1, f"Mix samples requires monophonic audio but C={wavs.size(1)}"
        wavs = wavs.mean(dim=1, keepdim=False)
        B, T = wavs.shape
        k = int(mix_p * B)
        mixed_sources_idx = torch.randperm(B)[:k]
        mixed_targets_idx = torch.randperm(B)[:k]
        aug_wavs = snr_mix(
            wavs[mixed_sources_idx],
            wavs[mixed_targets_idx],
            snr_low,
            snr_high,
            min_overlap,
        )
                descriptions = [info.description for info in infos]
        aug_infos = []
        for i, j in zip(mixed_sources_idx, mixed_targets_idx):
            text = mix_text(descriptions[i], descriptions[j])
            m = replace(infos[i])
            m.description = text
            aug_infos.append(m)

                aug_wavs = aug_wavs.unsqueeze(1)
        assert aug_wavs.shape[0] > 0, "Samples mixing returned empty batch."
        assert aug_wavs.dim() == 3, f"Returned wav should be [B, C, T] but dim = {aug_wavs.dim()}"
        assert aug_wavs.shape[0] == len(aug_infos), "Mismatch between number of wavs and infos in the batch"

        return aug_wavs, aug_infos      else:
                        B, C, T = wavs.shape
        k = int(mix_p * B)
        wav_idx = torch.randperm(B)[:k]
        wavs = wavs[wav_idx]
        infos = [infos[i] for i in wav_idx]
        assert wavs.shape[0] == len(infos), "Mismatch between number of wavs and infos in the batch"

        return wavs, infos  