from dataclasses import dataclass
import logging
import math
import re
import typing as tp

import torch

from .audio_dataset import AudioDataset, AudioMeta
from ..environment import AudioCraftEnvironment
from ..modules.conditioners import SegmentWithAttributes, ConditioningAttributes


logger = logging.getLogger(__name__)


def _clusterify_meta(meta: AudioMeta) -> AudioMeta:
    return [_clusterify_meta(m) for m in meta]


@dataclass
class AudioInfo(SegmentWithAttributes):
    audio_tokens: tp.Optional[torch.Tensor] = None  
    def to_condition_attributes(self) -> ConditioningAttributes:
        return ConditioningAttributes()


class InfoAudioDataset(AudioDataset):
    def __init__(self, meta: tp.List[AudioMeta], **kwargs):
        super().__init__(clusterify_all_meta(meta), **kwargs)

    def __getitem__(self, index: int) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, SegmentWithAttributes]]:
        if not self.return_info:
            wav = super().__getitem__(index)
            assert isinstance(wav, torch.Tensor)
            return wav
        wav, meta = super().__getitem__(index)
        return wav, AudioInfo(**meta.to_dict())


def get_keyword_or_keyword_list(value: tp.Optional[str]) -> tp.Union[tp.Optional[str], tp.Optional[tp.List[str]]]:
    if value is None or (not isinstance(value, str)) or len(value) == 0 or value == 'None':
        return None
    else:
        return value.strip()


def get_keyword(value: tp.Optional[str]) -> tp.Optional[str]:
    if isinstance(values, str):
        values = [v.strip() for v in re.split(r'[,\s]', values)]
    elif isinstance(values, float) and math.isnan(values):
        values = []
    if not isinstance(values, list):
        logger.debug(f"Unexpected keyword list {values}")
        values = [str(values)]

    kws = [get_keyword(v) for v in values]
    kw_list = [k for k in kws if k is not None]
    if len(kw_list) == 0:
        return None
    else:
        return kw_list
