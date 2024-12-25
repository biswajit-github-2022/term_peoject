
import logging
import math
import typing as tp
import torch
import numpy as np

from ..utils import utils
from ..modules.conditioners import (
    ClassifierFreeGuidanceDropout,
    ConditioningAttributes,
    ConditionType,
)
from .lm import LMModel

logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


class MagnetLMModel(LMModel):
    def __init__(self, subcodes_context: int = 5, compression_model_framerate: int = 50,
                 segment_duration: int = 10, span_len: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.causal = kwargs['causal']
        self.subcodes_context = subcodes_context
        self.span_len = span_len
        self._build_attn_masks(compression_model_framerate=compression_model_framerate,
                               segment_duration=segment_duration,
                               num_heads=kwargs['num_heads'],
                               device=kwargs['device'], dtype=kwargs['dtype'])

    def restricted_context_attn_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
                queries_pos = torch.arange(seq_len, device=device).view(-1, 1)
        keys_pos = torch.arange(seq_len, device=device).view(1, -1)

        delta = queries_pos - keys_pos
        valid = torch.abs(delta) <= self.subcodes_context
        return torch.where(
            valid,
            torch.zeros([], device=device, dtype=dtype),
            torch.full([], float('-inf'), device=device, dtype=dtype))

    def _stage_attn_mask(self, stage: int, seq_len: int, num_heads: int,
                         device: torch.device, dtype: torch.dtype) -> tp.Optional[torch.Tensor]:
        sa_mask = None

        if stage > 0 and self.subcodes_context > -1:
                        sa_mask = self.restricted_context_attn_mask(seq_len, device=device, dtype=dtype)

        if sa_mask is not None:
                        sa_mask = sa_mask.repeat((1, num_heads, 1, 1))

                        MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR = 8
            seq_len_aligned = \
                int(np.ceil(seq_len / MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR)) * MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR

            sa_mask_aligned = torch.zeros((1, num_heads, seq_len_aligned, seq_len_aligned), device=device, dtype=dtype)
            sa_mask_aligned[..., :seq_len, :seq_len] = sa_mask
            sa_mask = sa_mask_aligned

        return sa_mask

    def _build_attn_masks(self, compression_model_framerate: int, segment_duration: int, num_heads: int,
                          device: torch.device, dtype: torch.dtype):
        seq_len = compression_model_framerate * segment_duration
        self.attn_mask_per_stage = [self._stage_attn_mask(stage, seq_len, num_heads,
                                                          device, dtype) for stage in range(self.n_q)]

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
                 **kwargs) -> torch.Tensor:

        assert cfg_coef is None, "Unsupported in MAGNeT. Use max_cfg_coef,min_cfg_coef instead."
        assert two_step_cfg is None, "MAGNeT currently doesn't support two step classifier-free-guidance."
        assert remove_prompts is False, "MAGNeT currently doesn't support the remove_prompts arg."
        assert check is False, "MAGNeT currently doesn't support the check arg."
        assert cfg_coef_beta is None, "MAGNeT currently doesn't support the cfg_coef_beta arg."
                return self._generate_magnet(prompt=prompt,
                                     conditions=conditions,
                                     num_samples=num_samples,
                                     max_gen_len=max_gen_len,
                                     use_sampling=use_sampling,
                                     temp=temp,
                                     top_k=top_k,
                                     top_p=top_p,
                                     callback=callback, **kwargs)

    @torch.no_grad()
    def _generate_magnet(self,
                         prompt: tp.Optional[torch.Tensor] = None,
                         conditions: tp.List[ConditioningAttributes] = [],
                         num_samples: tp.Optional[int] = None,
                         max_gen_len: int = 256,
                         use_sampling: bool = True,
                         temp: float = 3.0,
                         top_k: int = 0,
                         top_p: float = 0.9,
                         callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                         max_cfg_coef: float = 10.0,
                         min_cfg_coef: float = 1.0,
                         decoding_steps: tp.List[int] = [20, 10, 10, 10],
                         anneal_temp: bool = True,
                         span_scoring='max',
                         span_arrangement='nonoverlap') -> torch.Tensor:
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

                                cfg_conditions: tp.Optional[ConditionTensors]
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            conditions = conditions + null_conditions
            tokenized = self.condition_provider.tokenize(conditions)
            cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, prompt_length = prompt.shape
        start_offset = prompt_length
        assert start_offset < max_gen_len

        mask_id = self.special_token_id

                shape = (B, K, max_gen_len)

        gen_codes = torch.full(shape, mask_id, dtype=torch.long, device=device)
                gen_codes[..., :start_offset] = prompt
                gen_sequence = gen_codes

        curr_step = 0
        for stage, n_steps in zip(range(self.n_q), decoding_steps):
            gen_sequence, curr_step = self._generate_stage(gen_sequence,
                                                           cfg_conditions,
                                                           stage=stage,
                                                           device=device,
                                                           prompt_length=prompt_length,
                                                           prompt=prompt,
                                                           temp=temp,
                                                           max_cfg_coef=max_cfg_coef,
                                                           min_cfg_coef=min_cfg_coef,
                                                           top_k=top_k,
                                                           top_p=top_p,
                                                           timesteps=n_steps,
                                                           anneal_temp=anneal_temp,
                                                           span_scoring=span_scoring,
                                                           use_sampling=use_sampling,
                                                           span_arrangement=span_arrangement,
                                                           curr_step=curr_step,
                                                           total_steps=sum(decoding_steps),
                                                           callback=callback)

        return gen_sequence

    @torch.no_grad()
    def _generate_stage(self,
                        gen_sequence: torch.Tensor,
                        condition_tensors: tp.Optional[ConditionTensors],
                        stage: int,
                        device: torch.device,
                        prompt_length: int = 0,
                        prompt: tp.Optional[torch.Tensor] = None,
                        use_sampling: bool = True,
                        temp: float = 3.0,
                        max_cfg_coef: float = 10.0,
                        min_cfg_coef: float = 1.0,
                        top_k: int = 0,
                        top_p: float = 0.0,
                        timesteps: int = 10,
                        anneal_temp: bool = True,
                        span_scoring: str = 'max',
                        span_arrangement: str = 'nonoverlap',
                        curr_step: int = 0,
                        total_steps: int = 0,
                        callback: tp.Optional[tp.Callable[[int, int], None]] = None) -> tp.Tuple[torch.Tensor, int]:
        B, K, T = gen_sequence.shape
        shape = (B, 1, T)  
        mask_id = self.special_token_id
        stage_gen_seq = torch.full(shape, mask_id, dtype=torch.long, device=device)

        assert span_arrangement == 'nonoverlap' or span_arrangement == 'stride1'
        chunk_masking = self.span_len > 1 and span_arrangement == 'nonoverlap'

        DONT_REMASK_ME_SCORE = -1e4

        model = self if self._fsdp is None else self._fsdp

        if chunk_masking:
                        n_chunks = T // self.span_len
            if T % self.span_len != 0:
                                T = self.span_len * n_chunks
                gen_sequence = gen_sequence[..., :T]
                stage_gen_seq = stage_gen_seq[..., :T]

            chunked_shape = (B, 1, n_chunks)
            n_prompt_chunks = prompt_length // self.span_len
            scores = torch.zeros(chunked_shape, dtype=torch.float32, device=device)
            scores[..., :n_prompt_chunks] = DONT_REMASK_ME_SCORE
            num_chunks_to_gen = n_chunks - n_prompt_chunks
        else:
                        scores = torch.zeros(shape, dtype=torch.float32, device=device)
            scores[..., :prompt_length] = DONT_REMASK_ME_SCORE
            gen_T = T - prompt_length

                for timestep, steps_left in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):

            mask_p = torch.cos(timestep * math.pi * 0.5)

            if chunk_masking:
                num_masked = max(int((mask_p * num_chunks_to_gen).item()), 1)
            else:
                num_masked = max(int((mask_p * gen_T).item()), 1)

                        run_lps_masking = (span_arrangement == 'stride1') and self.span_len > 1
            if run_lps_masking:
                                mask = torch.concat((
                    [self._least_probable_span_masking(scores[[i], :, :], num_masked).to(device)
                     for i in range(B)]), dim=0)
                stage_gen_seq[mask] = mask_id
            else:
                                masked = scores.topk(num_masked, dim=-1).indices
                if chunk_masking:
                    chunks_mask = torch.full(chunked_shape, False, dtype=torch.bool, device=device)
                    chunks_mask = chunks_mask.scatter(2, masked, True)
                    mask = torch.repeat_interleave(chunks_mask, self.span_len, dim=-1)
                    stage_gen_seq[mask] = mask_id
                else:
                    stage_gen_seq = stage_gen_seq.scatter(2, masked, mask_id)

            if prompt is not None:
                stage_gen_seq[..., :prompt_length] = prompt[:, stage, :].unsqueeze(1)

            gen_sequence[:, [stage], :] = stage_gen_seq
            if condition_tensors:
                                sequence = torch.cat([gen_sequence, gen_sequence], dim=0)

            all_logits = model(sequence, [], condition_tensors, stage=stage)

            if condition_tensors:
                                cond_logits, uncond_logits = all_logits.split(B, dim=0)                  clsfg_coef = float(mask_p) * max_cfg_coef + (1 - float(mask_p)) * min_cfg_coef
                logits = uncond_logits + (cond_logits - uncond_logits) * clsfg_coef
            else:
                logits = all_logits

                        t = temp * (steps_left / timesteps) if anneal_temp else temp

                        logits = logits[:, stage, :, :].unsqueeze(1)
            probs = torch.softmax(logits / max(t, 1e-2), dim=-1)
            if use_sampling:
                if top_p > 0.0:
                    sampled_tokens = utils.sample_top_p(probs, p=top_p)
                elif top_k > 0:
                    sampled_tokens = utils.sample_top_k(probs, k=top_k)
                else:
                    sampled_tokens = utils.multinomial(probs, num_samples=1)
            else:
                sampled_tokens = torch.argmax(logits, dim=-1, keepdim=True)

                        mask = stage_gen_seq == mask_id
            stage_gen_seq = torch.where(mask, sampled_tokens[..., 0], stage_gen_seq)
            gen_sequence[:, [stage], :] = stage_gen_seq

                        sampled_probs = torch.gather(probs, 3, sampled_tokens)[..., 0]

                        if chunk_masking:
                if span_scoring == 'max':
                                        scores = 1 - torch.max(sampled_probs.reshape((B, 1, n_chunks, -1)), dim=-1)[0]
                elif span_scoring == 'prod':
                                        scores = torch.sum(-torch.log(sampled_probs).reshape((B, 1, n_chunks, -1)), dim=-1)
                else:
                    raise NotImplementedError
            else:
                                scores = -torch.log(sampled_probs)

                        if chunk_masking:
                scores = scores.masked_fill(~chunks_mask, DONT_REMASK_ME_SCORE)
            else:
                scores = scores.masked_fill(~mask, DONT_REMASK_ME_SCORE)

            if callback is not None:
                curr_step += 1
                callback(curr_step, total_steps)

        return gen_sequence, curr_step

    def _construct_spans_mask(self, span_starts: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((1, 1, T), False, device=device)
        mask[:, :, span_starts] = True
        shifted_mask = mask.clone()
        for _ in range(self.span_len - 1):
            shifted_mask = torch.concat((torch.full((1, 1, 1), False, device=device), shifted_mask[:, :, :-1]), dim=-1)
            mask = torch.logical_or(mask, shifted_mask)
        return mask

    def _least_probable_span_masking(self, scores: torch.Tensor, num_masked_trg: int) -> torch.Tensor:
        T = scores.shape[-1]
        device = scores.device
        scores_unfolded = scores.unfold(2, self.span_len, 1)
                span_scores = scores_unfolded.sum(dim=-1)
        spans_by_scores = torch.argsort(span_scores[0, 0], descending=True)

        num_masked_trg = max(num_masked_trg, self.span_len)

                        min_u = num_masked_trg // self.span_len
        max_u = num_masked_trg - self.span_len + 1
        mid = round(0.5 * (min_u + max_u))

        if mid == min_u or mid == max_u:
            return self._construct_spans_mask(spans_by_scores[:mid], T, device)

        while mid > min_u and mid < max_u:
            mask = self._construct_spans_mask(spans_by_scores[:mid], T, device)
            n_masked = mask.sum()
            if n_masked > num_masked_trg:
                max_u = mid
                mid = round(0.5 * (min_u + max_u))
            else:
                min_u = mid
                mid = round(0.5 * (min_u + max_u))

        return mask
