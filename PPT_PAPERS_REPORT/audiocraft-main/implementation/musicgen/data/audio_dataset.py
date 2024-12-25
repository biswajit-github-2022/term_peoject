import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, fields
from contextlib import ExitStack
from functools import lru_cache
import gzip
import json
import logging
import os
from pathlib import Path
import random
import sys
import typing as tp

import torch
import torch.nn.functional as F

from .audio import audio_read, audio_info
from .audio_utils import convert_audio
from .zip import PathInZip

try:
    import dora
except ImportError:
    dora = None  

@dataclass(order=True)
class BaseInfo:

    @classmethod
    def _dict2fields(cls, dictionary: dict):
        return {
            field.name: dictionary[field.name]
            for field in fields(cls) if field.name in dictionary
        }

    @classmethod
    def from_dict(cls, dictionary: dict):
        _dictionary = cls._dict2fields(dictionary)
        return cls(**_dictionary)

    def to_dict(self):
        return {
            field.name: self.__getattribute__(field.name)
            for field in fields(self)
            }


@dataclass(order=True)
class AudioMeta(BaseInfo):
    path: str
    duration: float
    sample_rate: int
    amplitude: tp.Optional[float] = None
    weight: tp.Optional[float] = None
        info_path: tp.Optional[PathInZip] = None

    @classmethod
    def from_dict(cls, dictionary: dict):
        base = cls._dict2fields(dictionary)
        if 'info_path' in base and base['info_path'] is not None:
            base['info_path'] = PathInZip(base['info_path'])
        return cls(**base)

    def to_dict(self):
        d = super().to_dict()
        if d['info_path'] is not None:
            d['info_path'] = str(d['info_path'])
        return d


@dataclass(order=True)
class SegmentInfo(BaseInfo):
    meta: AudioMeta
    seek_time: float
            n_frames: int          total_frames: int      sample_rate: int       channels: int      

DEFAULT_EXTS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

logger = logging.getLogger(__name__)


def _get_audio_meta(file_path: str, minimal: bool = True) -> AudioMeta:
    info = audio_info(file_path)
    amplitude: tp.Optional[float] = None
    if not minimal:
        wav, sr = audio_read(file_path)
        amplitude = wav.abs().max().item()
    return AudioMeta(file_path, info.duration, info.sample_rate, amplitude)


def _resolve_audio_meta(m: AudioMeta, fast: bool = True) -> AudioMeta:
    def is_abs(m):
        if fast:
            return str(m)[0] == '/'
        else:
            os.path.isabs(str(m))

    if not dora:
        return m

    if not is_abs(m.path):
        m.path = dora.git_save.to_absolute_path(m.path)
    if m.info_path is not None and not is_abs(m.info_path.zip_path):
        m.info_path.zip_path = dora.git_save.to_absolute_path(m.path)
    return m


def find_audio_files(path: tp.Union[Path, str],
                     exts: tp.List[str] = DEFAULT_EXTS,
                     resolve: bool = True,
                     minimal: bool = True,
                     progress: bool = False,
                     workers: int = 0) -> tp.List[AudioMeta]:
    audio_files = []
    futures: tp.List[Future] = []
    pool: tp.Optional[ThreadPoolExecutor] = None
    with ExitStack() as stack:
        if workers > 0:
            pool = ThreadPoolExecutor(workers)
            stack.enter_context(pool)

        if progress:
            print("Finding audio files...")
        for root, folders, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = Path(root) / file
                if full_path.suffix.lower() in exts:
                    audio_files.append(full_path)
                    if pool is not None:
                        futures.append(pool.submit(_get_audio_meta, str(audio_files[-1]), minimal))
                    if progress:
                        print(format(len(audio_files), " 8d"), end='\r', file=sys.stderr)

        if progress:
            print("Getting audio metadata...")
        meta: tp.List[AudioMeta] = []
        for idx, file_path in enumerate(audio_files):
            try:
                if pool is None:
                    m = _get_audio_meta(str(file_path), minimal)
                else:
                    m = futures[idx].result()
                if resolve:
                    m = _resolve_audio_meta(m)
            except Exception as err:
                print("Error with", str(file_path), err, file=sys.stderr)
                continue
            meta.append(m)
            if progress:
                print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta


def load_audio_meta(path: tp.Union[str, Path],
                    resolve: bool = True, fast: bool = True) -> tp.List[AudioMeta]:
    open_fn = gzip.open if str(path).lower().endswith('.gz') else open
    with open_fn(path, 'rb') as fp:          lines = fp.readlines()
    meta = []
    for line in lines:
        d = json.loads(line)
        m = AudioMeta.from_dict(d)
        if resolve:
            m = _resolve_audio_meta(m, fast=fast)
        meta.append(m)
    return meta


def save_audio_meta(path: tp.Union[str, Path], meta: tp.List[AudioMeta]):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    open_fn = gzip.open if str(path).lower().endswith('.gz') else open
    with open_fn(path, 'wb') as fp:          for m in meta:
            json_str = json.dumps(m.to_dict()) + '\n'
            json_bytes = json_str.encode('utf-8')
            fp.write(json_bytes)


class AudioDataset:
    def __init__(self,
                 meta: tp.List[AudioMeta],
                 segment_duration: tp.Optional[float] = None,
                 shuffle: bool = True,
                 num_samples: int = 10_000,
                 sample_rate: int = 48_000,
                 channels: int = 2,
                 pad: bool = True,
                 sample_on_duration: bool = True,
                 sample_on_weight: bool = True,
                 min_segment_ratio: float = 0.5,
                 max_read_retry: int = 10,
                 return_info: bool = False,
                 min_audio_duration: tp.Optional[float] = None,
                 max_audio_duration: tp.Optional[float] = None,
                 shuffle_seed: int = 0,
                 load_wav: bool = True,
                 permutation_on_files: bool = False,
                 ):
        assert len(meta) > 0, "No audio meta provided to AudioDataset. Please check loading of audio meta."
        assert segment_duration is None or segment_duration > 0
        assert segment_duration is None or min_segment_ratio >= 0
        self.segment_duration = segment_duration
        self.min_segment_ratio = min_segment_ratio
        self.max_audio_duration = max_audio_duration
        self.min_audio_duration = min_audio_duration
        if self.min_audio_duration is not None and self.max_audio_duration is not None:
            assert self.min_audio_duration <= self.max_audio_duration
        self.meta: tp.List[AudioMeta] = self._filter_duration(meta)
        assert len(self.meta)          self.total_duration = sum(d.duration for d in self.meta)

        if segment_duration is None:
            num_samples = len(self.meta)
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.channels = channels
        self.pad = pad
        self.sample_on_weight = sample_on_weight
        self.sample_on_duration = sample_on_duration
        self.sampling_probabilities = self._get_sampling_probabilities()
        self.max_read_retry = max_read_retry
        self.return_info = return_info
        self.shuffle_seed = shuffle_seed
        self.current_epoch: tp.Optional[int] = None
        self.load_wav = load_wav
        if not load_wav:
            assert segment_duration is not None
        self.permutation_on_files = permutation_on_files
        if permutation_on_files:
            assert not self.sample_on_duration
            assert not self.sample_on_weight
            assert self.shuffle

    def start_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __len__(self):
        return self.num_samples

    def _get_sampling_probabilities(self, normalized: bool = True):
        This is only called if `segment_duration` is not None.

        You must use the provided random number generator `rng` for reproducibility.
        You can further make use of the index accessed.
        if AudioDataset has return_info=True in order to properly collate
        the samples of a batch.
        orig_len = len(meta)

                if self.min_audio_duration is not None:
            meta = [m for m in meta if m.duration >= self.min_audio_duration]

                if self.max_audio_duration is not None:
            meta = [m for m in meta if m.duration <= self.max_audio_duration]

        filtered_len = len(meta)
        removed_percentage = 100*(1-float(filtered_len)/orig_len)
        msg = 'Removed %.2f percent of the data because it was too short or too long.' % removed_percentage
        if removed_percentage < 10:
            logging.debug(msg)
        else:
            logging.warning(msg)
        return meta

    @classmethod
    def from_meta(cls, root: tp.Union[str, Path], **kwargs):
        root = Path(root)
        if root.is_dir():
            if (root / 'data.jsonl').exists():
                root = root / 'data.jsonl'
            elif (root / 'data.jsonl.gz').exists():
                root = root / 'data.jsonl.gz'
            else:
                raise ValueError("Don't know where to read metadata from in the dir. "
                                 "Expecting either a data.jsonl or data.jsonl.gz file but none found.")
        meta = load_audio_meta(root)
        return cls(meta, **kwargs)

    @classmethod
    def from_path(cls, root: tp.Union[str, Path], minimal_meta: bool = True,
                  exts: tp.List[str] = DEFAULT_EXTS, **kwargs):
        root = Path(root)
        if root.is_file():
            meta = load_audio_meta(root, resolve=True)
        else:
            meta = find_audio_files(root, exts, minimal=minimal_meta, resolve=True)
        return cls(meta, **kwargs)


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='audio_dataset',
        description='Generate .jsonl files by scanning a folder.')
    parser.add_argument('root', help='Root folder with all the audio files')
    parser.add_argument('output_meta_file',
                        help='Output file to store the metadata, ')
    parser.add_argument('--complete',
                        action='store_false', dest='minimal', default=True,
                        help='Retrieve all metadata, even the one that are expansive '
                             'to compute (e.g. normalization).')
    parser.add_argument('--resolve',
                        action='store_true', default=False,
                        help='Resolve the paths to be absolute and with no symlinks.')
    parser.add_argument('--workers',
                        default=10, type=int,
                        help='Number of workers.')
    args = parser.parse_args()
    meta = find_audio_files(args.root, DEFAULT_EXTS, progress=True,
                            resolve=args.resolve, minimal=args.minimal, workers=args.workers)
    save_audio_meta(args.output_meta_file, meta)


if __name__ == '__main__':
    main()
