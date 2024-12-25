

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
import re
import typing as tp
import unicodedata
import uuid

import dora
import torch

from ...data.audio import audio_read, audio_write


logger = logging.getLogger(__name__)


@dataclass
class ReferenceSample:
    id: str
    path: str
    duration: float


@dataclass
class Sample:
    id: str
    path: str
    epoch: int
    duration: float
    conditioning: tp.Optional[tp.Dict[str, tp.Any]]
    prompt: tp.Optional[ReferenceSample]
    reference: tp.Optional[ReferenceSample]
    generation_args: tp.Optional[tp.Dict[str, tp.Any]]

    def __hash__(self):
        return hash(self.id)

    def audio(self) -> tp.Tuple[torch.Tensor, int]:
        return audio_read(self.path)

    def audio_prompt(self) -> tp.Optional[tp.Tuple[torch.Tensor, int]]:
        return audio_read(self.prompt.path) if self.prompt is not None else None

    def audio_reference(self) -> tp.Optional[tp.Tuple[torch.Tensor, int]]:
        return audio_read(self.reference.path) if self.reference is not None else None


class SampleManager:
    def __init__(self, xp: dora.XP, map_reference_to_sample_id: bool = False):
        self.xp = xp
        self.base_folder: Path = xp.folder / xp.cfg.generate.path
        self.reference_folder = self.base_folder / 'reference'
        self.map_reference_to_sample_id = map_reference_to_sample_id
        self.samples: tp.List[Sample] = []
        self._load_samples()

    @property
    def latest_epoch(self):
        jsons = self.base_folder.glob('**/*.json')
        with ThreadPoolExecutor(6) as pool:
            self.samples = list(pool.map(self._load_sample, jsons))

    @staticmethod
    @lru_cache(2**26)
    def _load_sample(json_file: Path) -> Sample:
        with open(json_file, 'r') as f:
            data: tp.Dict[str, tp.Any] = json.load(f)
                prompt_data = data.get('prompt')
        prompt = ReferenceSample(id=prompt_data['id'], path=prompt_data['path'],
                                 duration=prompt_data['duration']) if prompt_data else None
                reference_data = data.get('reference')
        reference = ReferenceSample(id=reference_data['id'], path=reference_data['path'],
                                    duration=reference_data['duration']) if reference_data else None
                return Sample(id=data['id'], path=data['path'], epoch=data['epoch'], duration=data['duration'],
                      prompt=prompt, conditioning=data.get('conditioning'), reference=reference,
                      generation_args=data.get('generation_args'))

    def _init_hash(self):
        return hashlib.sha1()

    def _get_tensor_id(self, tensor: torch.Tensor) -> str:
        hash_id = self._init_hash()
        hash_id.update(tensor.numpy().data)
        return hash_id.hexdigest()

    def _get_sample_id(self, index: int, prompt_wav: tp.Optional[torch.Tensor],
                       conditions: tp.Optional[tp.Dict[str, str]]) -> str:
                        if prompt_wav is None and not conditions:
            return f"noinput_{uuid.uuid4().hex}"

                hr_label = ""
                hash_id = self._init_hash()
        hash_id.update(f"{index}".encode())
        if prompt_wav is not None:
            hash_id.update(prompt_wav.numpy().data)
            hr_label += "_prompted"
        else:
            hr_label += "_unprompted"
        if conditions:
            encoded_json = json.dumps(conditions, sort_keys=True).encode()
            hash_id.update(encoded_json)
            cond_str = "-".join([f"{key}={slugify(value)}"
                                 for key, value in sorted(conditions.items())])
            cond_str = cond_str[:100]              cond_str = cond_str if len(cond_str) > 0 else "unconditioned"
            hr_label += f"_{cond_str}"
        else:
            hr_label += "_unconditioned"

        return hash_id.hexdigest() + hr_label

    def _store_audio(self, wav: torch.Tensor, stem_path: Path, overwrite: bool = False) -> Path:
        existing_paths = [
            path for path in stem_path.parent.glob(stem_path.stem + '.*')
            if path.suffix != '.json'
        ]
        exists = len(existing_paths) > 0
        if exists and overwrite:
            logger.warning(f"Overwriting existing audio file with stem path {stem_path}")
        elif exists:
            return existing_paths[0]

        audio_path = audio_write(stem_path, wav, **self.xp.cfg.generate.audio)
        return audio_path

    def add_sample(self, sample_wav: torch.Tensor, epoch: int, index: int = 0,
                   conditions: tp.Optional[tp.Dict[str, str]] = None, prompt_wav: tp.Optional[torch.Tensor] = None,
                   ground_truth_wav: tp.Optional[torch.Tensor] = None,
                   generation_args: tp.Optional[tp.Dict[str, tp.Any]] = None) -> Sample:
        sample_id = self._get_sample_id(index, prompt_wav, conditions)
        reuse_id = self.map_reference_to_sample_id
        prompt, ground_truth = None, None
        if prompt_wav is not None:
            prompt_id = sample_id if reuse_id else self._get_tensor_id(prompt_wav.sum(0, keepdim=True))
            prompt_duration = prompt_wav.shape[-1] / self.xp.cfg.sample_rate
            prompt_path = self._store_audio(prompt_wav, self.base_folder / str(epoch) / 'prompt' / prompt_id)
            prompt = ReferenceSample(prompt_id, str(prompt_path), prompt_duration)
        if ground_truth_wav is not None:
            ground_truth_id = sample_id if reuse_id else self._get_tensor_id(ground_truth_wav.sum(0, keepdim=True))
            ground_truth_duration = ground_truth_wav.shape[-1] / self.xp.cfg.sample_rate
            ground_truth_path = self._store_audio(ground_truth_wav, self.base_folder / 'reference' / ground_truth_id)
            ground_truth = ReferenceSample(ground_truth_id, str(ground_truth_path), ground_truth_duration)
        sample_path = self._store_audio(sample_wav, self.base_folder / str(epoch) / sample_id, overwrite=True)
        duration = sample_wav.shape[-1] / self.xp.cfg.sample_rate
        sample = Sample(sample_id, str(sample_path), epoch, duration, conditions, prompt, ground_truth, generation_args)
        self.samples.append(sample)
        with open(sample_path.with_suffix('.json'), 'w') as f:
            json.dump(asdict(sample), f, indent=2)
        return sample

    def add_samples(self, samples_wavs: torch.Tensor, epoch: int,
                    conditioning: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None,
                    prompt_wavs: tp.Optional[torch.Tensor] = None,
                    ground_truth_wavs: tp.Optional[torch.Tensor] = None,
                    generation_args: tp.Optional[tp.Dict[str, tp.Any]] = None) -> tp.List[Sample]:
        samples = []
        for idx, wav in enumerate(samples_wavs):
            prompt_wav = prompt_wavs[idx] if prompt_wavs is not None else None
            gt_wav = ground_truth_wavs[idx] if ground_truth_wavs is not None else None
            conditions = conditioning[idx] if conditioning is not None else None
            samples.append(self.add_sample(wav, epoch, idx, conditions, prompt_wav, gt_wav, generation_args))
        return samples

    def get_samples(self, epoch: int = -1, max_epoch: int = -1, exclude_prompted: bool = False,
                    exclude_unprompted: bool = False, exclude_conditioned: bool = False,
                    exclude_unconditioned: bool = False) -> tp.Set[Sample]:
        if max_epoch >= 0:
            samples_epoch = max(sample.epoch for sample in self.samples if sample.epoch <= max_epoch)
        else:
            samples_epoch = self.latest_epoch if epoch < 0 else epoch
        samples = {
            sample
            for sample in self.samples
            if (
                (sample.epoch == samples_epoch) and
                (not exclude_prompted or sample.prompt is None) and
                (not exclude_unprompted or sample.prompt is not None) and
                (not exclude_conditioned or not sample.conditioning) and
                (not exclude_unconditioned or sample.conditioning)
            )
        }
        return samples


def slugify(value: tp.Any, allow_unicode: bool = False):
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def _match_stable_samples(samples_per_xp: tp.List[tp.Set[Sample]]) -> tp.Dict[str, tp.List[Sample]]:
        stable_samples_per_xp = [{
        sample.id: sample for sample in samples
        if sample.prompt is not None or sample.conditioning
    } for samples in samples_per_xp]
        stable_ids = {id for samples in stable_samples_per_xp for id in samples.keys()}
        stable_samples = {id: [xp.get(id) for xp in stable_samples_per_xp] for id in stable_ids}
            return {id: tp.cast(tp.List[Sample], samples) for id, samples in stable_samples.items() if None not in samples}


def _match_unstable_samples(samples_per_xp: tp.List[tp.Set[Sample]]) -> tp.Dict[str, tp.List[Sample]]:
        unstable_samples_per_xp = [[
        sample for sample in sorted(samples, key=lambda x: x.id)
        if sample.prompt is None and not sample.conditioning
    ] for samples in samples_per_xp]
        min_len = min([len(samples) for samples in unstable_samples_per_xp])
    unstable_samples_per_xp = [samples[:min_len] for samples in unstable_samples_per_xp]
        return {
        f'noinput_{i}': [samples[i] for samples in unstable_samples_per_xp] for i in range(min_len)
    }


def get_samples_for_xps(xps: tp.List[dora.XP], **kwargs) -> tp.Dict[str, tp.List[Sample]]:
    managers = [SampleManager(xp) for xp in xps]
    samples_per_xp = [manager.get_samples(**kwargs) for manager in managers]
    stable_samples = _match_stable_samples(samples_per_xp)
    unstable_samples = _match_unstable_samples(samples_per_xp)
    return dict(stable_samples, **unstable_samples)
