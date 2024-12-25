
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
import typing as tp

import flashy
import omegaconf
import torch
from torch import nn

from .. import optim
from ..optim import fsdp
from ..utils import checkpoint
from ..utils.autocast import TorchAutocast
from ..utils.best_state import BestStateDictManager
from ..utils.deadlock import DeadlockDetect
from ..utils.profiler import Profiler
from ..utils.utils import copy_state, dict_from_config, model_hash, with_rank_rng


class StandardSolver(ABC, flashy.BaseSolver):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.logger.info(f"Instantiating solver {self.__class__.__name__} for XP {self.xp.sig}")
        self.logger.info(f"All XP logs are stored in {self.xp.folder}")
        self.cfg = cfg
        self.device = cfg.device
        self.model: nn.Module
        self._continue_best_source_keys = ['best_state', 'fsdp_best_state']
        self._fsdp_modules: tp.List[fsdp.FSDP] = []
        self._ema_sources: nn.ModuleDict = nn.ModuleDict()
        self.ema: tp.Optional[optim.ModuleDictEMA] = None
        self.dataloaders: tp.Dict[str, torch.utils.data.DataLoader] = dict()
        self._log_updates = self.cfg.logging.get('log_updates', 10)
        if self.cfg.logging.log_tensorboard:
            self.init_tensorboard(**self.cfg.get('tensorboard'))
        if self.cfg.logging.log_wandb and self:
            self.init_wandb(**self.cfg.get('wandb'))
                        dtype_best: tp.Optional[torch.dtype] = None
        if self.cfg.fsdp.use:
            dtype_best = getattr(torch, self.cfg.fsdp.param_dtype)              assert isinstance(dtype_best, torch.dtype)
        elif self.cfg.autocast:
            dtype_best = getattr(torch, self.cfg.autocast_dtype)              assert isinstance(dtype_best, torch.dtype)
        self.best_state: BestStateDictManager = BestStateDictManager(dtype=dtype_best)
                self.fsdp_best_state: tp.Dict[str, tp.Any] = {}
        self.register_stateful('best_state', 'fsdp_best_state')          self._new_best_state: bool = False                  self.build_dataloaders()
        if self.cfg.execute_only is None:
            assert 'train' in self.dataloaders, "The train dataset split must be provided."
            assert 'valid' in self.dataloaders, "The valid dataset split must be provided."
        self.train_updates_per_epoch = len(self.dataloaders['train']) if 'train' in self.dataloaders else 0
        if self.cfg.optim.updates_per_epoch:
            self.train_updates_per_epoch = self.cfg.optim.updates_per_epoch
        self.total_updates = self.train_updates_per_epoch * self.cfg.optim.epochs
                self.build_model()
        self.logger.info("Model hash: %s", model_hash(self.model))
        assert 'model' in self.stateful.sources, \
            "Please register the model to stateful with self.register_stateful('model') in build_model."
        self.profiler = Profiler(self.model, **self.cfg.profiler)
        self.initialize_ema()
        self.register_stateful('ema')
        assert self.ema is None or 'ema' in self.stateful.sources, \
            "Please register the ema to stateful with self.register_stateful('ema') in build_model."
        self.deadlock_detect = DeadlockDetect(**self.cfg.deadlock)
                model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
                        mem_usage = model_size * 4 * 4 / 1000
        self.logger.info("Model size: %.2f M params", model_size)
        self.logger.info("Base memory usage, with model, grad and optim: %.2f GB", mem_usage)

    @property
    def autocast(self):
        used on the stage for best state identification (most likely, `valid`). If None, then
        no best state is saved.
        latest states. The best state will be used at evaluation stages instead of the latest states.

        Shortcut around `BestStateDictManager.register` method. You can pass any number of
        attribute, included nested attributes and those will be included into the checkpoints
        and automatically restored when `BaseSolver.restore` is called.

        The registered sources are used to instantiate a ModuleDictEMA instance.
        The ModuleDictEMA keeps a `nn.ModuleDict` module that is updated when self.ema.step() is called
        and swapped with the original state sources with self.swap_ema_state() method.

        Usage:
            self.register_ema('model')
        on the `BestStateDictManager.update` method to update the best state_dict with latest weights
        if the registered states happen to match to the best performing setup.
        self.logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
        self.logger.info("Size: %.1f MB", mb)

    @abstractmethod
    def build_model(self):
        EMA object is created if the optim.ema.model.decay value is non-null.
        ...

    @abstractmethod
    def show(self):
        is_sharded = self.cfg.fsdp.use
        if not flashy.distrib.is_rank_zero() and not is_sharded:
            return
        self.logger.info("Model hash: %s", model_hash(self.model))
        state = self.state_dict()
        epoch = self.epoch - 1  
                if self.cfg.checkpoint.save_every:
            if epoch % self.cfg.checkpoint.save_every == 0:
                minimal_state = state
                if self.cfg.checkpoint.keep_every_states is not None and len(self.cfg.checkpoint.keep_every_states) > 0:
                    minimal_state = {
                        name: source for name, source in state.items()
                        if name in self.cfg.checkpoint.keep_every_states
                    }
                epoch_checkpoint_path = self.epoch_checkpoint_path(epoch)
                checkpoint.save_checkpoint(minimal_state, epoch_checkpoint_path, is_sharded)

                if self.cfg.checkpoint.save_last:
            last_checkpoint_path = self.checkpoint_path()
            checkpoint.save_checkpoint(state, last_checkpoint_path, is_sharded)

                checkpoint.flush_stale_checkpoints(self.checkpoint_path())

    def load_from_pretrained(self, name: str) -> dict:
        raise NotImplementedError("Solver does not provide a way to load pretrained models.")

    def load_checkpoints(self, load_best: bool = False, ignore_state_keys: tp.List[str] = []) -> tp.Optional[dict]:
                is_sharded = self.cfg.fsdp.use
        load_from_path: tp.Optional[Path] = None
        checkpoint_source: tp.Optional[checkpoint.CheckpointSource] = None

        if load_best:
            self.logger.info("Trying to load state_dict from best state.")

        state: tp.Optional[dict] = None
        rank0_checkpoint_path = self.checkpoint_path(use_fsdp=False)
        current_checkpoint_path = self.checkpoint_path()
        _pretrained_prefix = '//pretrained/'
        continue_pretrained = (self.cfg.continue_from or '').startswith(_pretrained_prefix)
        if rank0_checkpoint_path.exists():
            self.logger.info(f"Loading existing checkpoint: {current_checkpoint_path}")
            load_from_path = current_checkpoint_path
            checkpoint.check_sharded_checkpoint(current_checkpoint_path, rank0_checkpoint_path)
            checkpoint_source = checkpoint.CheckpointSource.CURRENT_XP
        elif self.cfg.continue_from and not continue_pretrained:
            self.logger.info(f"Continuing from provided checkpoint: {self.cfg.continue_from}")
                        load_from_path = checkpoint.resolve_checkpoint_path(self.cfg.continue_from, use_fsdp=False)
            if load_from_path is None:
                self.logger.error('Could not resolve the continue_from checkpoint %s', self.cfg.continue_from)
                raise RuntimeError(f'Could not resolve continue_from checkpoint {self.cfg.continue_from}')
            checkpoint_source = checkpoint.CheckpointSource.OTHER

        if load_from_path is not None:
            state = checkpoint.load_checkpoint(load_from_path, is_sharded)
        elif continue_pretrained:
            self.logger.info("Loading a pretrained model. Ignoring 'load_best' and 'ignore_state_keys' params.")
            state = self.load_from_pretrained(self.cfg.continue_from[len(_pretrained_prefix):])
            checkpoint_source = checkpoint.CheckpointSource.PRETRAINED
            load_best = True

                if checkpoint_source is not None and checkpoint_source != checkpoint.CheckpointSource.CURRENT_XP:
            assert state is not None
            self.logger.info("Checkpoint source is not the current xp: Load state_dict from best state.")
            load_best = True
            state = {key: state[key] for key in self._continue_best_source_keys if key in state}
                                    if 'fsdp_best_state' in state and state['fsdp_best_state']:
                state.pop('best_state', None)
                self.logger.info("... Loaded checkpoint has FSDP best state")
                                    elif self.cfg.fsdp.use:
                if 'fsdp_best_state' not in state or not state['fsdp_best_state']:
                                        state['fsdp_best_state'] = state.pop('best_state')
                    self.logger.info("... Loaded checkpoint does not have FSDP best state. Use regular best state")

        if state is not None:
            if load_best:
                self.logger.info("Ignoring keys when loading best %r", ignore_state_keys)
                for key in set(ignore_state_keys):
                    if key in state:
                        state.pop(key)
                has_best_state = 'best_state' in state or 'fsdp_best_state' in state
                assert has_best_state, ("Trying to load best state but neither 'best_state'",
                                        " or 'fsdp_best_state' found in checkpoints.")
            self.load_state_dict(state)

                        epoch = float(self.epoch)
        avg_epoch = flashy.distrib.average_metrics({'epoch': epoch})['epoch']
        if avg_epoch != epoch:
            raise RuntimeError(
                f"Inconsistent loading of checkpoints happened, our epoch is {epoch} "
                f"but average of epochs is {avg_epoch}, at least one gpu must have a "
                "different epoch number.")

                        if load_best:
            self.logger.info("Loading state_dict from best state.")
            if not self.cfg.fsdp.use and self.fsdp_best_state:
                                self.logger.info("... Loading from FSDP best state dict.")
                self.best_state.load_state_dict(self.fsdp_best_state)

                        if self.cfg.fsdp.use:
                self.logger.info("FSDP is used, loading from FSDP best state.")
                with fsdp.switch_to_full_state_dict(self._fsdp_modules):
                                        self.load_state_dict(self.fsdp_best_state)
            else:
                                self._load_new_state_dict(self.best_state.state_dict())

                                    if self.ema is not None:
                self.logger.info("Re-initializing EMA from best state")
                self.initialize_ema()

            if self.cfg.fsdp.use:
                self.logger.info("Re-initializing best state after using FSDP best state.")
                for name in self.best_state.states.keys():
                    state_source = self._get_state_source(name)
                    self.best_state.update(name, state_source)

        return state

    def restore(self, load_best: bool = False, replay_metrics: bool = False,
                ignore_state_keys: tp.List[str] = []) -> bool:
        self.logger.info("Restoring weights and history.")
        restored_checkpoints = self.load_checkpoints(load_best, ignore_state_keys)

        self.logger.info("Model hash: %s", model_hash(self.model))

        if replay_metrics and len(self.history) > 0:
            self.logger.info("Replaying past metrics...")
            for epoch, stages in enumerate(self.history):
                for stage_name, metrics in stages.items():
                                                            self.result_logger._log_summary(stage_name, metrics, step=epoch + 1, step_name='epoch',
                                                    formatter=self.get_formatter(stage_name))
        return restored_checkpoints is not None

    def commit(self, save_checkpoints: bool = True):

        Metrics for a given stage are stored in _pending_metrics and committed by the solver afterwards.
        Children solvers can extend this method with custom behavior, e.g.:

            def run_epoch(self):
                ...                 super().run_epoch()
                ...         assert len(self.state_dict()) > 0
        self.restore(replay_metrics=True)          self.log_hyperparams(dict_from_config(self.cfg))
        for epoch in range(self.epoch, self.cfg.optim.epochs + 1):
            if self.should_stop_training():
                return
            self.run_epoch()
                        self.commit()

    def should_stop_training(self) -> bool:
        stage_every = self.cfg[stage_name].get('every', None)
        is_last_epoch = self.epoch == self.cfg.optim.epochs
        is_epoch_every = (stage_every and self.epoch % stage_every == 0)
        return is_last_epoch or is_epoch_every

    @abstractmethod
    def run_step(self, idx: int, batch: tp.Any, metrics: dict):
        self.model.train(self.is_training)

        loader = self.dataloaders[dataset_split]
                if flashy.distrib.world_size() > 1 \
           and isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
            loader.sampler.set_epoch(self.epoch)
        updates_per_epoch = self.train_updates_per_epoch if self.is_training else len(loader)
        if self.cfg.benchmark_no_load:
            self.logger.warning("Fake loading for benchmarking: re-using first batch")
            batch = next(iter(loader))
            loader = [batch] * updates_per_epoch          lp = self.log_progress(self.current_stage, loader, total=updates_per_epoch, updates=self.log_updates)
        average = flashy.averager()          instant_average = flashy.averager()          metrics: dict = {}

        with self.profiler, self.deadlock_detect:              for idx, batch in enumerate(lp):
                self.deadlock_detect.update('batch')
                if idx >= updates_per_epoch:
                    break
                metrics = {}
                metrics = self.run_step(idx, batch, metrics)
                self.deadlock_detect.update('step')
                                if self.ema is not None and self.is_training and (idx + 1) % self.cfg.optim.ema.updates == 0:
                    self.logger.debug("EMA model step")
                    self.ema.step()
                self.deadlock_detect.update('ema')
                self.profiler.step()
                instant_metrics = instant_average(metrics)
                if lp.update(**instant_metrics):
                    instant_average = flashy.averager()                  metrics = average(metrics)                  self.deadlock_detect.update('end_batch')

        metrics = flashy.distrib.average_metrics(metrics, updates_per_epoch)
        return metrics

    def train(self):
        return self.common_train_valid('valid')

    @abstractmethod
    def evaluate(self):
        ...

    def run_one_stage(self, stage_name: str):
        fn = {
            'generate': with_rank_rng()(self.generate),
            'evaluate': self.evaluate,
            'valid': self.valid,
        }
        if stage_name not in fn:
            raise ValueError(f'Trying to run stage {stage_name} is not supported.')
        assert len(self.state_dict()) > 0
        self._start_epoch()
        with torch.no_grad(), self.swap_best_state():
            self.run_stage(stage_name, fn[stage_name])
        if not self.cfg.execute_inplace:
            self.commit(save_checkpoints=False)

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
