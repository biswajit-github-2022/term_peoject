
from dataclasses import dataclass
from functools import partial
import logging
import math
import typing as tp

import torch
from torch import nn

from ..utils import utils
from ..modules.streaming import StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
    _drop_description_condition
)
from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.activations import get_activation_fn


logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
        std = 1 / math.sqrt(input_dim)
        if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std          return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(m: nn.Module,
               method: str,
               init_depth: tp.Optional[int] = None,
               zero_bias_init: bool = False):
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)


class ScaledEmbedding(nn.Embedding):
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
            logits: torch.Tensor      mask: torch.Tensor  

class LMModel(StreamingModule):
    def __init__(self, pattern_provider: CodebooksPatternProvider, condition_provider: ConditioningProvider,
                 fuser: ConditionFuser, n_q: int = 8, card: int = 1024, dim: int = 128, num_heads: int = 8,
                 hidden_scale: int = 4, norm: str = 'layer_norm', norm_first: bool = False,
                 emb_lr: tp.Optional[float] = None, bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None, depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}, two_step_cfg: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.card = card
        embed_dim = self.card + 1
        self.n_q = n_q
        self.dim = dim
        self.pattern_provider = pattern_provider
        self.two_step_cfg = two_step_cfg
        self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first, **kwargs)
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)
        self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None

    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        for linear in self.linears:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

    @property
    def special_token_id(self) -> int:
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def forward(self, sequence: torch.Tensor,
                conditions: tp.List[ConditioningAttributes],
                condition_tensors: tp.Optional[ConditionTensors] = None,
                stage: int = -1) -> torch.Tensor:
        B, K, S = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])
        if condition_tensors is None:
            assert not self._is_streaming, "Conditions tensors should be precomputed when streaming."
                        conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
                        condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions, "Shouldn't pass both conditions and condition_tensors."

        input_, cross_attention_input = self.fuser(input_, condition_tensors)

        out = self.transformer(input_, cross_attention_src=cross_attention_input,
                               src_mask=(self.attn_mask_per_stage[stage] if stage >= 0 else None))          if self.out_norm:
            out = self.out_norm(out)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  
                if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:]

        return logits  
    def compute_predictions(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None,
            stage: int = -1,
            keep_only_valid_steps: bool = True) -> LMOutput:
        B, K, T = codes.shape
        codes = codes.contiguous()
                pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=keep_only_valid_steps,
        )

                model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes, conditions, condition_tensors, stage=stage)                          logits = logits.permute(0, 3, 1, 2)                  logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=keep_only_valid_steps
        )
        logits = logits.permute(0, 2, 3, 1)          logits_mask = logits_mask[None, :, :].expand(B, -1, -1)          return LMOutput(logits, logits_mask)

    def _sample_next_token(self,
                           sequence: torch.Tensor,
                           cfg_conditions: CFGConditions,
                           unconditional_state: State,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None,
                           cfg_coef_beta: tp.Optional[float] = None,
                           two_step_cfg: tp.Optional[bool] = None) -> torch.Tensor:
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
        if cfg_coef_beta is not None:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                                                sequence = torch.cat([sequence, sequence, sequence], dim=0)
            all_logits = model(
                sequence,
                conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits, wav_logits, uncond_logits = all_logits.split(B, dim=0)                  logits = uncond_logits + cfg_coef * (
                    wav_logits + cfg_coef_beta * (cond_logits - wav_logits) - uncond_logits
                    )

        elif two_step_cfg and cfg_conditions != {}:
            assert isinstance(cfg_conditions, tuple), type(cfg_conditions)
            condition_tensors, null_condition_tensors = cfg_conditions
            cond_logits = model(sequence, conditions=[], condition_tensors=condition_tensors)
            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            uncond_logits = model(sequence, conditions=[], condition_tensors=null_condition_tensors)
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_coef
        else:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                                sequence = torch.cat([sequence, sequence], dim=0)
            all_logits = model(
                sequence,
                conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits, uncond_logits = all_logits.split(B, dim=0)                  logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
            else:
                logits = all_logits

        logits = logits.permute(0, 1, 3, 2)          logits = logits[..., -1]  
                if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token

    @torch.no_grad()
    def generate(self,
                 prompt: tp.Optional[torch.Tensor] = None,
                 conditions: tp.List[ConditioningAttributes] = [],
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 top_k: int = 250,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 cfg_coef_beta: tp.Optional[float] = None,
                 two_step_cfg: tp.Optional[bool] = None,
                 remove_prompts: bool = False,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                 ) -> torch.Tensor:
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

                possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]

                                                                                cfg_conditions: CFGConditions
        cfg_conditions = {}
        if cfg_coef_beta is not None:
            if conditions:
                wav_conditions = _drop_description_condition(conditions)
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                conditions = conditions + wav_conditions + null_conditions
                tokenized = self.condition_provider.tokenize(conditions)
                cfg_conditions = self.condition_provider(tokenized)
        elif conditions:
            two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
            if conditions:
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                if two_step_cfg:
                    cfg_conditions = (
                        self.condition_provider(self.condition_provider.tokenize(conditions)),
                        self.condition_provider(self.condition_provider.tokenize(null_conditions)),
                    )
                else:
                    conditions = conditions + null_conditions
                    tokenized = self.condition_provider.tokenize(conditions)
                    cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        start_offset = T
        assert start_offset < max_gen_len

        pattern = self.pattern_provider.get_pattern(max_gen_len)
                unknown_token = -1

                gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
                gen_codes[..., :start_offset] = prompt
                gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
                        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        with self.streaming():
            unconditional_state = self.get_streaming_state()
            prev_offset = 0
            gen_sequence_len = gen_sequence.shape[-1]              for offset in range(start_offset_sequence, gen_sequence_len):
                                curr_sequence = gen_sequence[..., prev_offset:offset]
                curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
                if check:
                                        assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.special_token_id)).all()
                                        assert not (curr_sequence == unknown_token).any()
                                next_token = self._sample_next_token(
                    curr_sequence, cfg_conditions, unconditional_state, use_sampling, temp, top_k, top_p,
                    cfg_coef=cfg_coef, cfg_coef_beta=cfg_coef_beta, two_step_cfg=two_step_cfg)
                                                valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
                next_token[~valid_mask] = self.special_token_id
                                                gen_sequence[..., offset:offset+1] = torch.where(
                    gen_sequence[..., offset:offset+1] == unknown_token,
                    next_token, gen_sequence[..., offset:offset+1]
                )
                prev_offset = offset
                if callback is not None:
                    callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)
        unconditional_state.clear()

                assert not (gen_sequence == unknown_token).any()
                        assert (
            gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id)
        ).all()
                out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)

                assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_gen_len]

                assert (out_codes >= 0).all() and (out_codes <= self.card).all()
        return out_codes
