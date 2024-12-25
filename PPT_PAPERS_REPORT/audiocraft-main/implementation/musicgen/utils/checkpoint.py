
from enum import Enum
import logging
from pathlib import Path
import re
import typing as tp

import flashy
import torch

from ..environment import AudioCraftEnvironment


logger = logging.getLogger(__name__)


class CheckpointSource(Enum):
    CURRENT_XP = "current_xp"
    PRETRAINED = "pretrained"
    OTHER = "other"


def checkpoint_name(name: tp.Optional[str] = None, rank: tp.Optional[int] = None, use_fsdp: bool = False) -> str:
    suffix = ''
    if rank is None:
        rank = flashy.distrib.rank()
    if rank > 0 and use_fsdp:
        suffix = '.' + str(rank)
    name_part = ''
    if name is not None:
        name_part = f'_{name}'
    return f'checkpoint{name_part}.th{suffix}'


def is_sharded_checkpoint(path: Path) -> bool:

    Args:
        sig_or_path (Path or str): Checkpoint path or dora signature.
        name (str, optional): Name suffix for the checkpoint file stem.
        rank (optional, int): Rank for distributed processing, retrieved with flashy if not provided.
        use_fsdp (bool): Whether the calling solver relies on FSDP.
    Returns:
        Path, optional: Resolved checkpoint path, if it exists.
    if is_sharded:
        rank0_checkpoint_path = checkpoint_path.parent / checkpoint_name(use_fsdp=False)
        if rank0_checkpoint_path.exists():
            check_sharded_checkpoint(checkpoint_path, rank0_checkpoint_path)
    state = torch.load(checkpoint_path, 'cpu')
    logger.info("Checkpoint loaded from %s", checkpoint_path)
    return state


def save_checkpoint(state: tp.Any, checkpoint_path: Path, is_sharded: bool = False) -> None:
    if keep_last is None or keep_last <= 0:
        return
    checkpoint_dir = checkpoint_path.parent
    suffix = ''
    if flashy.distrib.rank() > 0:
        suffix = f'.{flashy.distrib.rank()}'
    checkpoint_files_with_epoch = []
    for path in Path(checkpoint_dir).glob(f'checkpoint_*.th{suffix}'):
        epoch_part = path.name.split('.', 1)[0].split('_', 1)[1]
        if epoch_part.isdigit():
            checkpoint_files_with_epoch.append((path, int(epoch_part)))
    checkpoint_files = [path for path, _ in list(sorted(checkpoint_files_with_epoch, key=lambda t: t[1]))]
    total_to_flush = max(0, len(checkpoint_files) - keep_last)
    files_to_flush = checkpoint_files[:total_to_flush]
    for path in files_to_flush:
        logger.debug("Removing checkpoint: %s", str(path))
        path.unlink(missing_ok=True)


def check_sharded_checkpoint(checkpoint_path: Path, rank0_checkpoint_path: Path) -> None:
    def _barrier_if_sharded():
        if is_sharded:
            flashy.distrib.barrier()

    if flashy.distrib.is_rank_zero():
        token = Path(str(checkpoint_path) + '.tmp.done')
        if token.exists():
            token.unlink()
    _barrier_if_sharded()
    with flashy.utils.write_and_rename(checkpoint_path) as f:
        torch.save(state, f)
        _barrier_if_sharded()
        if flashy.distrib.is_rank_zero():
            token.touch()
        _barrier_if_sharded()
    _barrier_if_sharded()
    if flashy.distrib.rank() == 0:
        token.unlink()
