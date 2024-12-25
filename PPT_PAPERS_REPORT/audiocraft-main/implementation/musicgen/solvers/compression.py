
import logging
import multiprocessing
from pathlib import Path
import typing as tp

import flashy
import omegaconf
import torch
from torch import nn

from . import base, builders
from .. import models, quantization
from ..utils import checkpoint
from ..utils.samples.manager import SampleManager
from ..utils.utils import get_pool_executor


logger = logging.getLogger(__name__)


class CompressionSolver(base.StandardSolver):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.rng: torch.Generator          self.adv_losses = builders.get_adversarial_losses(self.cfg)
        self.aux_losses = nn.ModuleDict()
        self.info_losses = nn.ModuleDict()
        assert not cfg.fsdp.use, "FSDP not supported by CompressionSolver."
        loss_weights = dict()
        for loss_name, weight in self.cfg.losses.items():
            if loss_name in ['adv', 'feat']:
                for adv_name, _ in self.adv_losses.items():
                    loss_weights[f'{loss_name}_{adv_name}'] = weight
            elif weight > 0:
                self.aux_losses[loss_name] = builders.get_loss(loss_name, self.cfg)
                loss_weights[loss_name] = weight
            else:
                self.info_losses[loss_name] = builders.get_loss(loss_name, self.cfg)
        self.balancer = builders.get_balancer(loss_weights, self.cfg.balancer)
        self.register_stateful('adv_losses')

    @property
    def best_metric_name(self) -> tp.Optional[str]:
                return None

    def build_model(self):
        self.dataloaders = builders.get_audio_datasets(self.cfg)

    def show(self):
        x = batch.to(self.device)
        y = x.clone()

        qres = self.model(x)
        assert isinstance(qres, quantization.QuantizedResult)
        y_pred = qres.x
                metrics['bandwidth'] = qres.bandwidth.mean()

        if self.is_training:
            d_losses: dict = {}
            if len(self.adv_losses) > 0 and torch.rand(1, generator=self.rng).item() <= 1 / self.cfg.adversarial.every:
                for adv_name, adversary in self.adv_losses.items():
                    disc_loss = adversary.train_adv(y_pred, y)
                    d_losses[f'd_{adv_name}'] = disc_loss
                metrics['d_loss'] = torch.sum(torch.stack(list(d_losses.values())))
            metrics.update(d_losses)

        balanced_losses: dict = {}
        other_losses: dict = {}

                if qres.penalty is not None and qres.penalty.requires_grad:
            other_losses['penalty'] = qres.penalty  
                for adv_name, adversary in self.adv_losses.items():
            adv_loss, feat_loss = adversary(y_pred, y)
            balanced_losses[f'adv_{adv_name}'] = adv_loss
            balanced_losses[f'feat_{adv_name}'] = feat_loss

                for loss_name, criterion in self.aux_losses.items():
            loss = criterion(y_pred, y)
            balanced_losses[loss_name] = loss

                metrics.update(balanced_losses)
        metrics.update(other_losses)
        metrics.update(qres.metrics)

        if self.is_training:
                        other_loss = torch.tensor(0., device=self.device)
            if 'penalty' in other_losses:
                other_loss += other_losses['penalty']
            if other_loss.requires_grad:
                other_loss.backward(retain_graph=True)
                ratio1 = sum(p.grad.data.norm(p=2).pow(2)
                             for p in self.model.parameters() if p.grad is not None)
                assert isinstance(ratio1, torch.Tensor)
                metrics['ratio1'] = ratio1.sqrt()

                                    metrics['g_loss'] = self.balancer.backward(balanced_losses, y_pred)
                        metrics.update(self.balancer.metrics)
            ratio2 = sum(p.grad.data.norm(p=2).pow(2)
                         for p in self.model.parameters() if p.grad is not None)
            assert isinstance(ratio2, torch.Tensor)
            metrics['ratio2'] = ratio2.sqrt()

                        flashy.distrib.sync_model(self.model)
            if self.cfg.optim.max_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.optim.max_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

                info_losses: dict = {}
        with torch.no_grad():
            for loss_name, criterion in self.info_losses.items():
                loss = criterion(y_pred, y)
                info_losses[loss_name] = loss

        metrics.update(info_losses)

                adv_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith('adv')]
        if len(adv_losses) > 0:
            metrics['adv'] = torch.sum(torch.stack(adv_losses))
        feat_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith('feat')]
        if len(feat_losses) > 0:
            metrics['feat'] = torch.sum(torch.stack(feat_losses))

        return metrics

    def run_epoch(self):
                self.rng = torch.Generator()
        self.rng.manual_seed(1234 + self.epoch)
                super().run_epoch()

    def evaluate(self):
        self.model.eval()
        sample_manager = SampleManager(self.xp, map_reference_to_sample_id=True)
        generate_stage_name = str(self.current_stage)

        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        for batch in lp:
            reference, _ = batch
            reference = reference.to(self.device)
            with torch.no_grad():
                qres = self.model(reference)
            assert isinstance(qres, quantization.QuantizedResult)

            reference = reference.cpu()
            estimate = qres.x.cpu()
            sample_manager.add_samples(estimate, self.epoch, ground_truth_wavs=reference)

        flashy.distrib.barrier()

    def load_from_pretrained(self, name: str) -> dict:
        model = models.CompressionModel.get_pretrained(name)
        if isinstance(model, models.DAC):
            raise RuntimeError("Cannot fine tune a DAC model.")
        elif isinstance(model, models.HFEncodecCompressionModel):
            self.logger.warning('Trying to automatically convert a HuggingFace model '
                                'to AudioCraft, this might fail!')
            state = model.model.state_dict()
            new_state = {}
            for k, v in state.items():
                if k.startswith('decoder.layers') and '.conv.' in k and '.block.' not in k:
                                        layer = int(k.split('.')[2])
                    if isinstance(model.model.decoder.layers[layer].conv, torch.nn.ConvTranspose1d):

                        k = k.replace('.conv.', '.convtr.')
                k = k.replace('encoder.layers.', 'encoder.model.')
                k = k.replace('decoder.layers.', 'decoder.model.')
                k = k.replace('conv.', 'conv.conv.')
                k = k.replace('convtr.', 'convtr.convtr.')
                k = k.replace('quantizer.layers.', 'quantizer.vq.layers.')
                k = k.replace('.codebook.', '._codebook.')
                new_state[k] = v
            state = new_state
        elif isinstance(model, models.EncodecModel):
            state = model.state_dict()
        else:
            raise RuntimeError(f"Cannot fine tune model type {type(model)}.")
        return {
            'best_state': {'model': state}
        }

    @staticmethod
    def model_from_checkpoint(checkpoint_path: tp.Union[Path, str],
                              device: tp.Union[torch.device, str] = 'cpu') -> models.CompressionModel:
        checkpoint_path = str(checkpoint_path)
        if checkpoint_path.startswith('//pretrained/'):
            name = checkpoint_path.split('/', 3)[-1]
            return models.CompressionModel.get_pretrained(name, device)
        logger = logging.getLogger(__name__)
        logger.info(f"Loading compression model from checkpoint: {checkpoint_path}")
        _checkpoint_path = checkpoint.resolve_checkpoint_path(checkpoint_path, use_fsdp=False)
        assert _checkpoint_path is not None, f"Could not resolve compression model checkpoint path: {checkpoint_path}"
        state = checkpoint.load_checkpoint(_checkpoint_path)
        assert state is not None and 'xp.cfg' in state, f"Could not load compression model from ckpt: {checkpoint_path}"
        cfg = state['xp.cfg']
        cfg.device = device
        compression_model = models.builders.get_compression_model(cfg).to(device)
        assert compression_model.sample_rate == cfg.sample_rate, "Compression model sample rate should match"

        assert 'best_state' in state and state['best_state'] != {}
        assert 'exported' not in state, "When loading an exported checkpoint, use the //pretrained/ prefix."
        compression_model.load_state_dict(state['best_state']['model'])
        compression_model.eval()
        logger.info("Compression model loaded!")
        return compression_model

    @staticmethod
    def wrapped_model_from_checkpoint(cfg: omegaconf.DictConfig,
                                      checkpoint_path: tp.Union[Path, str],
                                      device: tp.Union[torch.device, str] = 'cpu') -> models.CompressionModel:
        compression_model = CompressionSolver.model_from_checkpoint(checkpoint_path, device)
        compression_model = models.builders.get_wrapped_compression_model(compression_model, cfg)
        return compression_model


def evaluate_audio_reconstruction(y_pred: torch.Tensor, y: torch.Tensor, cfg: omegaconf.DictConfig) -> dict:
