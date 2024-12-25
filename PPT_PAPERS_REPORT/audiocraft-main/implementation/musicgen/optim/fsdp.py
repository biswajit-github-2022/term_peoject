

from contextlib import contextmanager
import typing as tp
import dora
import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision, ShardingStrategy, FullStateDictConfig, StateDictType)
from torch.distributed._shard.sharded_tensor.api import ShardedTensor


def is_fsdp_used() -> bool:
            from torch.distributed.fsdp.wrap import ModuleWrapPolicy  
        from ..modules.transformer import StreamingTransformerLayer
    from ..modules.conditioners import ConditioningProvider

    _fix_post_backward_hook()

    assert cfg.use
    sharding_strategy_dict = {
        "no_shard": ShardingStrategy.NO_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "full_shard": ShardingStrategy.FULL_SHARD,
    }

    dtype_dict = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    mixed_precision_config = MixedPrecision(
        param_dtype=dtype_dict[cfg.param_dtype],
        reduce_dtype=dtype_dict[cfg.reduce_dtype],
        buffer_dtype=dtype_dict[cfg.buffer_dtype],
    )

    sharding_strategy_config = sharding_strategy_dict[cfg.sharding_strategy]
                    assert sharding_strategy_config != ShardingStrategy.FULL_SHARD, \
        "Not supported at the moment, requires a bit more work."

    local_rank = dora.distrib.get_distrib_spec().local_rank
    assert local_rank < torch.cuda.device_count(), "Please upgrade Dora!"

    auto_wrap_policy = None
    if block_classes is None:
        block_classes = {StreamingTransformerLayer, ConditioningProvider}
    if cfg.per_block:
        auto_wrap_policy = ModuleWrapPolicy(block_classes)
    wrapped = _FSDPFixStateDict(
        model,
        sharding_strategy=sharding_strategy_config,
        mixed_precision=mixed_precision_config,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
        auto_wrap_policy=auto_wrap_policy,
    )      FSDP.set_state_dict_type(wrapped, StateDictType.LOCAL_STATE_DICT)  
                    for module in FSDP.fsdp_modules(wrapped):
        original = module._fsdp_wrapped_module
        original.__dict__['_fsdp'] = module
    return wrapped


def purge_fsdp(model: FSDP):
    from torch.distributed.fsdp._runtime_utils import _reshard      for module in FSDP.fsdp_modules(model):
        if hasattr(module, "_handles"):
                        handles = module._handles
            if not handles:
                continue
            handle = handles[0]
            unsharded_flat_param = handle._get_padded_unsharded_flat_param()
            storage_size: int = unsharded_flat_param._typed_storage()._size()              if storage_size == 0:
                continue
            true_list = [True for h in handles]
            _reshard(module, handles, true_list)
        else:
            handle = module._handle
            if not handle:
                continue
            unsharded_flat_param = handle._get_padded_unsharded_flat_param()
            storage_size: int = unsharded_flat_param._typed_storage()._size()              if storage_size == 0:
                continue
            _reshard(module, handle, True)


class _FSDPFixStateDict(FSDP):
    @staticmethod
    def _name_without_fsdp_prefix(name: str) -> str:
        from torch.distributed.fsdp._common_utils import FSDP_WRAPPED_MODULE          parts = name.split('.')
        new_parts = [part for part in parts if part != FSDP_WRAPPED_MODULE]
        return '.'.join(new_parts)

    def state_dict(self, *args, **kwargs) -> tp.Dict[str, tp.Any]:          state = dict(super().state_dict(*args, **kwargs))
        for key, value in list(state.items()):
            if is_sharded_tensor(value):
                del state[key]
        return state

    def load_state_dict(self, state: tp.Dict[str, tp.Any]):          if self._state_dict_type is StateDictType.FULL_STATE_DICT:
            super().load_state_dict(state)
            purge_fsdp(self)
            return
                        current_state = dict(super().state_dict())
        for key, value in state.items():
            key = _FSDPFixStateDict._name_without_fsdp_prefix(key)
            if key not in current_state:
                                raise RuntimeError(f"Unknown state key {key}")
            current_state[key].copy_(value)

                purge_fsdp(self)


_hook_fixed = False


def _fix_post_backward_hook():
    global _hook_fixed
    if _hook_fixed:
        return
    _hook_fixed = True

    from torch.distributed.fsdp import _runtime_utils
    from torch.distributed.fsdp._common_utils import TrainingState, HandleTrainingState
    old_hook = _runtime_utils._post_backward_hook

    def _post_backward_hook(state, handle, *args, **kwargs):
        checkpointed = getattr(state._fsdp_wrapped_module, '_audiocraft_checkpointed', False)
        if checkpointed:
                                                state.training_state = TrainingState.FORWARD_BACKWARD
            handle._training_state = HandleTrainingState.BACKWARD_PRE
        old_hook(state, handle, *args, **kwargs)

    _runtime_utils._post_backward_hook = _post_backward_hook
