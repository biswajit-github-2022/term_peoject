

from enum import Enum
import logging
import typing as tp

import dora
import flashy
import omegaconf
import torch
from torch import nn
from torch.optim import Optimizer

try:
    from torch.optim.lr_scheduler import LRScheduler  except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from .base import StandardSolver
from .. import adversarial, data, losses, metrics, optim
from ..utils.utils import dict_from_config, get_loader


logger = logging.getLogger(__name__)


class DatasetType(Enum):
    AUDIO = "audio"
    MUSIC = "music"
    SOUND = "sound"


def get_solver(cfg: omegaconf.DictConfig) -> StandardSolver:
    if defined for each modules, to create the different groups.

    Args:
        model (nn.Module): torch model
    Returns:
        List of parameter groups
    Supported optimizers: Adam, AdamW

    Args:
        params (nn.Module or iterable of torch.Tensor): Parameters to optimize.
        cfg (DictConfig): Optimization-related configuration.
    Returns:
        torch.optim.Optimizer.
    Supported learning rate schedulers: ExponentialLRScheduler, PlateauLRScheduler

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        cfg (DictConfig): Schedule-related configuration.
        total_updates (int): Total number of updates.
    Returns:
        torch.optim.Optimizer.

    Args:
        module_dict (nn.ModuleDict): ModuleDict for which to compute the EMA.
        cfg (omegaconf.DictConfig): Optim EMA configuration.
    Returns:
        optim.ModuleDictEMA: EMA version of the ModuleDict.
    klass = {
        'l1': torch.nn.L1Loss,
        'l2': torch.nn.MSELoss,
        'mel': losses.MelSpectrogramL1Loss,
        'mrstft': losses.MRSTFTLoss,
        'msspec': losses.MultiScaleMelSpectrogramLoss,
        'sisnr': losses.SISNR,
        'wm_detection': losses.WMDetectionLoss,
        'wm_mb': losses.WMMbLoss,
        'tf_loudnessratio': losses.TFLoudnessRatio
    }[loss_name]
    kwargs = dict(getattr(cfg, loss_name))
    return klass(**kwargs)


def get_balancer(loss_weights: tp.Dict[str, float], cfg: omegaconf.DictConfig) -> losses.Balancer:
    klass = {
        'msd': adversarial.MultiScaleDiscriminator,
        'mpd': adversarial.MultiPeriodDiscriminator,
        'msstftd': adversarial.MultiScaleSTFTDiscriminator,
    }[name]
    adv_cfg: tp.Dict[str, tp.Any] = dict(getattr(cfg, name))
    return klass(**adv_cfg)


def get_adversarial_losses(cfg) -> nn.ModuleDict:
    kwargs = dict_from_config(cfg)
    return metrics.ViSQOL(**kwargs)


def get_fad(cfg: omegaconf.DictConfig) -> metrics.FrechetAudioDistanceMetric:
    kld_metrics = {
        'passt': metrics.PasstKLDivergenceMetric,
    }
    klass = kld_metrics[cfg.model]
    kwargs = dict_from_config(cfg.get(cfg.model))
    return klass(**kwargs)


def get_text_consistency(cfg: omegaconf.DictConfig) -> metrics.TextConsistencyMetric:
    assert cfg.model == 'chroma_base', "Only support 'chroma_base' method for chroma cosine similarity metric"
    kwargs = dict_from_config(cfg.get(cfg.model))
    return metrics.ChromaCosineSimilarityMetric(**kwargs)


def get_audio_datasets(cfg: omegaconf.DictConfig,
                       dataset_type: DatasetType = DatasetType.AUDIO) -> tp.Dict[str, torch.utils.data.DataLoader]:
    dataloaders: dict = {}

    sample_rate = cfg.sample_rate
    channels = cfg.channels
    seed = cfg.seed
    max_sample_rate = cfg.datasource.max_sample_rate
    max_channels = cfg.datasource.max_channels

    assert cfg.dataset is not None, "Could not find dataset definition in config"

    dataset_cfg = dict_from_config(cfg.dataset)
    splits_cfg: dict = {}
    splits_cfg['train'] = dataset_cfg.pop('train')
    splits_cfg['valid'] = dataset_cfg.pop('valid')
    splits_cfg['evaluate'] = dataset_cfg.pop('evaluate')
    splits_cfg['generate'] = dataset_cfg.pop('generate')
    execute_only_stage = cfg.get('execute_only', None)

    for split, path in cfg.datasource.items():
        if not isinstance(path, str):
            continue          if execute_only_stage is not None and split != execute_only_stage:
            continue
        logger.info(f"Loading audio data split {split}: {str(path)}")
        assert (
            cfg.sample_rate <= max_sample_rate
        ), f"Expecting a max sample rate of {max_sample_rate} for datasource but {sample_rate} found."
        assert (
            cfg.channels <= max_channels
        ), f"Expecting a max number of channels of {max_channels} for datasource but {channels} found."

        split_cfg = splits_cfg[split]
        split_kwargs = {k: v for k, v in split_cfg.items()}
        kwargs = {**dataset_cfg, **split_kwargs}          kwargs['sample_rate'] = sample_rate
        kwargs['channels'] = channels

        if kwargs.get('permutation_on_files') and cfg.optim.updates_per_epoch:
            kwargs['num_samples'] = (
                flashy.distrib.world_size() * cfg.dataset.batch_size * cfg.optim.updates_per_epoch)

        num_samples = kwargs['num_samples']
        shuffle = kwargs['shuffle']

        return_info = kwargs.pop('return_info')
        batch_size = kwargs.pop('batch_size', None)
        num_workers = kwargs.pop('num_workers')

        if dataset_type == DatasetType.MUSIC:
            dataset = data.music_dataset.MusicDataset.from_meta(path, **kwargs)
        elif dataset_type == DatasetType.SOUND:
            dataset = data.sound_dataset.SoundDataset.from_meta(path, **kwargs)
        elif dataset_type == DatasetType.AUDIO:
            dataset = data.info_audio_dataset.InfoAudioDataset.from_meta(path, return_info=return_info, **kwargs)
        else:
            raise ValueError(f"Dataset type is unsupported: {dataset_type}")

        loader = get_loader(
            dataset,
            num_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            collate_fn=dataset.collater if return_info else None,
            shuffle=shuffle,
        )
        dataloaders[split] = loader

    return dataloaders
