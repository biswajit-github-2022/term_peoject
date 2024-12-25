

import typing as tp

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from xformers import ops

from .rope import RotaryEmbedding
from .streaming import StreamingModule

_efficient_attention_backend: str = 'torch'


def set_efficient_attention_backend(backend: str = 'torch'):
        global _efficient_attention_backend
    assert _efficient_attention_backend in ['xformers', 'torch']
    _efficient_attention_backend = backend


def _get_attention_time_dimension(memory_efficient: bool) -> int:
    if _efficient_attention_backend == 'torch' and memory_efficient:
        return 2
    else:
        return 1


def _is_profiled() -> bool:
        try:
        from xformers.profiler import profiler
    except ImportError:
        return False
    return profiler._Profiler._CURRENT_PROFILER is not None


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    if norm_type == 'layer_norm':
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
        assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)      phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def expand_repeated_kv(x: torch.Tensor, n_rep: int, memory_efficient: bool) -> torch.Tensor:
    This rescales diagonally the residual outputs close to 0, with a learnt scale.

    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): If True, expect `[*, C]` shaped tensors, otherwise, `[*, C, T]`.
        device (torch.device or str, optional): Device on which to initialize the module.
        dtype (torch.dtype, optional): dtype to use to initialize the module.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        dropout (float): Dropout level.
        bias (bool): Use bias in projections.
        causal (bool): Causal mask applied automatically.
        past_context (int, optional): Receptive field for the causal mask, infinite if None.
        custom (bool): Use custom MHA implementation, for testing / benchmarking.
        memory_efficient (bool): Use xformers based memory efficient attention.
        attention_as_float32 (bool): Perform the attention as float32
            (especially important with memory_efficient as autocast won't do this automatically).
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        cross_attention: Should be true when used as a cross attention.
            All keys and values must be available at once, streaming is only for the queries.
            Cannot be used with `causal` or `rope` (as it wouldn't make sens to
            interpret the time steps in the keys relative to those in the queries).
        safe_streaming (bool): Bug fix, will go away with xformers update.
        qk_layer_norm (bool): Layer normalization applied to queries and keys before dot product.
        kv_repeat (int): If > 1, will repeat keys and queries multiple times (need to divide num_heads).
            This will lead to faster decoding time on A100 or other GPUs with tensorcore.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    This also integrates cross_attention, when passing `cross_attention=True`,
    rather than having two separate classes like in PyTorch.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        dropout (float): Dropout both for MHA and FF.
        bias_ff (bool): Use bias for FF.
        bias_attn (bool): Use bias for MHA.
        causal (bool): Causal mask applied automatically.
        past_context (int, optional): Receptive field for the causal mask, infinite if None.
        custom (bool): Use custom MHA implementation, for testing / benchmarking.
        memory_efficient (bool): Use xformers based memory efficient attention.
        attention_as_float32 (bool): Perform the attention as float32
            (especially important with memory_efficient as autocast won't do this automatically).
        qk_layer_norm (bool): Layer normalization applied to queries and keys before dot product in attention.
        qk_layer_norm_cross (bool): Same for the cross attention.
        cross_attention (bool): If True, expect to get secondary input for cross-attention.
            Cross attention will use the default MHA, as it typically won't require
            special treatment.
        layer_scale (float, optional): If not None, LayerScale will be used with
            the given value as initial scale.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        attention_dropout (float, optional): If not None, separate the value of the dimension dropout
            in FFN and of the attention dropout.
        kv_repeat (int): If > 1, will repeat keys and queries multiple times (need to divide num_heads).
            This will lead to faster decoding time on A100 or other GPUs with tensorcore.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
        **kwargs: See `nn.TransformerEncoderLayer`.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        dropout (float): Dropout both for MHA and FF.
        bias_ff (bool): Use bias for FF.
        bias_attn (bool): Use bias for MHA.
        causal (bool): Causal mask applied automatically.
        past_context (int, optional): Receptive field for the causal mask, infinite if None.
        custom (bool): Use custom MHA implementation, for testing / benchmarking.
        memory_efficient (bool): Use xformers based memory efficient attention.
        attention_as_float32 (bool): Perform the attention as float32
            (especially important with memory_efficient as autocast won't do this automatically).
        cross_attention (bool): If True, expect to get secondary input for cross-attention.
        layer_scale (float, optional): If not None, LayerScale will be used
            with the given value as initial scale.
        positional_embedding (str): Positional embedding strategy (sin, rope, or sin_rope).
        max_period (float): Maximum period of the time embedding.
        positional_scale (float): Scale of positional embedding, set to 0 to deactivate.
        xpos (bool): Apply xpos exponential decay to positional embedding (rope only).
        lr (float, optional): learning rate override through the `make_optim_group` API.
        weight_decay (float, optional): Weight_decay override through the `make_optim_group` API.
        layer_class: (subclass of `StreamingTransformerLayer): class to use
            to initialize the layers, allowing further customization outside of AudioCraft.
        checkpointing (str): Checkpointing strategy to reduce memory usage.
            No checkpointing if set to 'none'. Per layer checkpointing using PyTorch
            if set to 'torch' (entire layer checkpointed, i.e. linears are evaluated twice,
            minimal memory usage, but maximal runtime). Finally, `xformers_default` provide
            a policy for opting-out some operations of the checkpointing like
            linear layers and attention, providing a middle ground between speed and memory.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
        **kwargs: See `nn.TransformerEncoderLayer`.
