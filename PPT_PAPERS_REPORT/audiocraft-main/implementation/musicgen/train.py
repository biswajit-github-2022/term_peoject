

import logging
import multiprocessing
import os
from pathlib import Path
import sys
import typing as tp

from dora import git_save, hydra_main, XP
import flashy
import hydra
import omegaconf

from .environment import AudioCraftEnvironment
from .utils.cluster import get_slurm_parameters

logger = logging.getLogger(__name__)


def resolve_config_dset_paths(cfg):

    Args:
        xp (XP): Dora experiment for which to retrieve the solver.
        override_cfg (dict or None): If not None, should be a dict used to
            override some values in the config of `xp`. This will not impact
            the XP signature or folder. The format is different
            than the one used in Dora grids, nested keys should actually be nested dicts,
            not flattened, e.g. `{'optim': {'batch_size': 32}}`.
        restore (bool): If `True` (the default), restore state from the last checkpoint.
        load_best (bool): If `True` (the default), load the best state from the checkpoint.
        ignore_state_keys (list[str]): List of sources to ignore when loading the state, e.g. `optimizer`.
        disable_fsdp (bool): if True, disables FSDP entirely. This will
            also automatically skip loading the EMA. For solver specific
            state sources, like the optimizer, you might want to
            use along `ignore_state_keys=['optimizer']`. Must be used with `load_best=True`.
    See `get_solver_from_xp` for more information.
