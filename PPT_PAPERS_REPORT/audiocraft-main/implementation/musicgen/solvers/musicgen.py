
from pathlib import Path
import time
import typing as tp
import warnings

import flashy
import math
import omegaconf
import torch
from torch.nn import functional as F

from . import base, builders
from .compression import CompressionSolver
from .. import metrics as eval_metrics
from .. import models
from ..data.audio_dataset import AudioDataset
from ..data.music_dataset import MusicDataset, MusicInfo, AudioInfo
from ..data.audio_utils import normalize_audio
from ..modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition, \
            StyleConditioner, _drop_description_condition
from ..utils.cache import CachedBatchWriter, CachedBatchLoader
from ..utils.samples.manager import SampleManager
from ..utils.utils import get_dataset_from_loader, is_jsonable, warn_once, model_hash


class MusicGenSolver(base.StandardSolver):
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.MUSIC

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
                self.generation_params = {
            'use_sampling': self.cfg.generate.lm.use_sampling,
            'temp': self.cfg.generate.lm.temp,
            'top_k': self.cfg.generate.lm.top_k,
            'top_p': self.cfg.generate.lm.top_p,
        }
        self._best_metric_name: tp.Optional[str] = 'ce'

        self._cached_batch_writer = None
        self._cached_batch_loader = None
        if cfg.cache.path:
            if cfg.cache.write:
                self._cached_batch_writer = CachedBatchWriter(Path(cfg.cache.path))
                if self.cfg.cache.write_num_shards:
                    self.logger.warning("Multiple shard cache, best_metric_name will be set to None.")
                    self._best_metric_name = None
            else:
                self._cached_batch_loader = CachedBatchLoader(
                    Path(cfg.cache.path), cfg.dataset.batch_size, cfg.dataset.num_workers,
                    min_length=self.cfg.optim.updates_per_epoch or 1)
                self.dataloaders['original_train'] = self.dataloaders['train']
                self.dataloaders['train'] = self._cached_batch_loader  
    @staticmethod
    def get_eval_solver_from_sig(sig: str, dtype: tp.Optional[str] = None,
                                 device: tp.Optional[str] = None, autocast: bool = True,
                                 batch_size: tp.Optional[int] = None,
                                 override_cfg: tp.Optional[tp.Union[dict, omegaconf.DictConfig]] = None,
                                 **kwargs):
        from audiocraft import train
        our_override_cfg: tp.Dict[str, tp.Any] = {'optim': {'ema': {'use': False}}}
        our_override_cfg['autocast'] = autocast
        if dtype is not None:
            our_override_cfg['dtype'] = dtype
        if device is not None:
            our_override_cfg['device'] = device
        if batch_size is not None:
            our_override_cfg['dataset'] = {'batch_size': batch_size}
        if override_cfg is None:
            override_cfg = {}
        override_cfg = omegaconf.OmegaConf.merge(
            omegaconf.DictConfig(override_cfg), omegaconf.DictConfig(our_override_cfg))          solver = train.get_solver_from_sig(
            sig, override_cfg=override_cfg,
            load_best=True, disable_fsdp=True,
            ignore_state_keys=['optimizer', 'ema'], **kwargs)
        solver.model.eval()
        return solver

    def get_formatter(self, stage_name: str) -> flashy.Formatter:
        return flashy.Formatter({
            'lr': '.2E',
            'ce': '.3f',
            'ppl': '.3f',
            'grad_norm': '.3E',
        }, exclude_keys=['ce_q*', 'ppl_q*'])

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        return self._best_metric_name

    def build_model(self) -> None:
        self.dataloaders = builders.get_audio_datasets(self.cfg, dataset_type=self.DATASET_TYPE)

    def show(self) -> None:
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
            check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        check_synchronization_points = idx == 1 and self.device == 'cuda'

        condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(
            batch, check_synchronization_points)

        self.deadlock_detect.update('tokens_and_conditions')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('warn')

        with self.autocast:
            style_mask = None
            if hasattr(self.model.condition_provider.conditioners, 'self_wav'):
                if isinstance(self.model.condition_provider.conditioners.self_wav, StyleConditioner):
                    style_mask = self.model.condition_provider.conditioners.self_wav.mask

            model_output = self.model.compute_predictions(audio_tokens, [], condition_tensors)              logits = model_output.logits
            if style_mask is not None:
                mask = padding_mask & model_output.mask & style_mask
            else:
                mask = padding_mask & model_output.mask
            ce, ce_per_codebook = self._compute_cross_entropy(logits, audio_tokens, mask)
            loss = ce
        self.deadlock_detect.update('loss')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('default')

        if self.is_training:
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            self.deadlock_detect.update('scale')
            if self.cfg.fsdp.use:
                loss.backward()
                flashy.distrib.average_tensors(self.model.buffers())
            elif self.cfg.optim.eager_sync:
                with flashy.distrib.eager_sync_model(self.model):
                    loss.backward()
            else:
                                                loss.backward()
                flashy.distrib.sync_model(self.model)
            self.deadlock_detect.update('backward')

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.cfg.optim.max_norm:
                if self.cfg.fsdp.use:
                    metrics['grad_norm'] = self.model.clip_grad_norm_(self.cfg.optim.max_norm)                  else:
                    metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optim.max_norm
                    )
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.deadlock_detect.update('optim')
            if self.scaler is not None:
                scale = self.scaler.get_scale()
                metrics['grad_scale'] = scale
            if not loss.isfinite().all():
                raise RuntimeError("Model probably diverged.")

        metrics['ce'] = ce
        metrics['ppl'] = torch.exp(ce)
        for k, ce_q in enumerate(ce_per_codebook):
            metrics[f'ce_q{k + 1}'] = ce_q
            metrics[f'ppl_q{k + 1}'] = torch.exp(ce_q)

        return metrics

    @torch.no_grad()
    def run_generate_step(self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
                          gen_duration: float, prompt_duration: tp.Optional[float] = None,
                          remove_text_conditioning: bool = False,
                          ) -> dict:
        bench_start = time.time()
        audio, meta = batch
        assert audio.size(0) == len(meta), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(meta)})"
        )
                attributes = [x.to_condition_attributes() for x in meta]
        if remove_text_conditioning:
            attributes = _drop_description_condition(attributes)

                if prompt_duration is None:
            prompt_audio = None
        else:
            assert prompt_duration < gen_duration, "Prompt duration must be lower than target generation duration"
            prompt_audio_frames = int(prompt_duration * self.compression_model.sample_rate)
            prompt_audio = audio[..., :prompt_audio_frames]

                if prompt_audio is None or prompt_audio.nelement() == 0:
            num_samples = len(attributes)
            prompt_tokens = None
        else:
            num_samples = None
            prompt_audio = prompt_audio.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt_audio)
            assert scale is None, "Compression model in MusicGen should not require rescaling."

                with self.autocast:
            total_gen_len = math.ceil(gen_duration * self.compression_model.frame_rate)
            gen_tokens = self.model.generate(
                prompt_tokens, attributes, max_gen_len=total_gen_len,
                num_samples=num_samples, **self.generation_params)

                assert gen_tokens.dim() == 3
        gen_audio = self.compression_model.decode(gen_tokens, None)

        bench_end = time.time()
        gen_outputs = {
            'rtf': (bench_end - bench_start) / gen_duration,
            'ref_audio': audio,
            'gen_audio': gen_audio,
            'gen_tokens': gen_tokens,
            'prompt_audio': prompt_audio,
            'prompt_tokens': prompt_tokens,
        }
        return gen_outputs

    def generate_audio(self) -> dict:
        self.model.eval()
        with torch.no_grad():
            return self.generate_audio()

    def run_epoch(self):
        if self.cfg.cache.write:
            if ((self.epoch - 1) % self.cfg.cache.write_num_shards) != self.cfg.cache.write_shard:
                return
        super().run_epoch()

    def train(self):
        if self._cached_batch_writer is not None:
            self._cached_batch_writer.start_epoch(self.epoch)
        if self._cached_batch_loader is None:
            dataset = get_dataset_from_loader(self.dataloaders['train'])
            assert isinstance(dataset, AudioDataset)
            dataset.current_epoch = self.epoch
        else:
            self._cached_batch_loader.start_epoch(self.epoch)
        return super().train()

    def evaluate_audio_generation(self) -> dict:
        self.model.eval()
        with torch.no_grad():
            metrics: dict = {}
            if self.cfg.evaluate.metrics.base:
                metrics.update(self.common_train_valid('evaluate'))
            gen_metrics = self.evaluate_audio_generation()
            return {**metrics, **gen_metrics}
