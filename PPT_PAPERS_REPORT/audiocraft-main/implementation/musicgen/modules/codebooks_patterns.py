
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
import logging
import typing as tp

from abc import ABC, abstractmethod
import torch

LayoutCoord = namedtuple('LayoutCoord', ['t', 'q'])  PatternLayout = tp.List[tp.List[LayoutCoord]]  logger = logging.getLogger(__name__)


@dataclass
class Pattern:
                    layout: PatternLayout
    timesteps: int
    n_q: int

    def __post_init__(self):
        assert len(self.layout) > 0
        self._validate_layout()
        self._build_reverted_sequence_scatter_indexes = lru_cache(100)(self._build_reverted_sequence_scatter_indexes)
        self._build_pattern_sequence_scatter_indexes = lru_cache(100)(self._build_pattern_sequence_scatter_indexes)
        logger.info("New pattern, time steps: %d, sequence steps: %d", self.timesteps, len(self.layout))

    def _validate_layout(self):
        q_timesteps = {q: 0 for q in range(self.n_q)}
        for s, seq_coords in enumerate(self.layout):
            if len(seq_coords) > 0:
                qs = set()
                for coord in seq_coords:
                    qs.add(coord.q)
                    last_q_timestep = q_timesteps[coord.q]
                    assert coord.t >= last_q_timestep, \
                        f"Past timesteps are found in the sequence for codebook = {coord.q} at step {s}"
                    q_timesteps[coord.q] = coord.t
                                assert len(qs) == len(seq_coords), \
                    f"Multiple entries for a same codebook are found at step {s}"

    @property
    def num_sequence_steps(self):
        return len(self.layout) - 1

    @property
    def max_delay(self):
        max_t_in_seq_coords = 0
        for seq_coords in self.layout[1:]:
            for coords in seq_coords:
                max_t_in_seq_coords = max(max_t_in_seq_coords, coords.t + 1)
        return max_t_in_seq_coords - self.timesteps

    @property
    def valid_layout(self):
        valid_step = len(self.layout) - self.max_delay
        return self.layout[:valid_step]

    def starts_with_special_token(self):
        return self.layout[0] == []

    def get_sequence_coords_with_timestep(self, t: int, q: tp.Optional[int] = None):
        assert t <= self.timesteps, "provided timesteps is greater than the pattern's number of timesteps"
        if q is not None:
            assert q <= self.n_q, "provided number of codebooks is greater than the pattern's number of codebooks"
        coords = []
        for s, seq_codes in enumerate(self.layout):
            for code in seq_codes:
                if code.t == t and (q is None or code.q == q):
                    coords.append((s, code))
        return coords

    def get_steps_with_timestep(self, t: int, q: tp.Optional[int] = None) -> tp.List[int]:
        return [step for step, coords in self.get_sequence_coords_with_timestep(t, q)]

    def get_first_step_with_timesteps(self, t: int, q: tp.Optional[int] = None) -> tp.Optional[int]:
        steps_with_timesteps = self.get_steps_with_timestep(t, q)
        return steps_with_timesteps[0] if len(steps_with_timesteps) > 0 else None

    def _build_pattern_sequence_scatter_indexes(self, timesteps: int, n_q: int, keep_only_valid_steps: bool,
                                                device: tp.Union[torch.device, str] = 'cpu'):
        assert n_q == self.n_q, f"invalid number of codebooks for the sequence and the pattern: {n_q} != {self.n_q}"
        assert timesteps <= self.timesteps, "invalid number of timesteps used to build the sequence from the pattern"
                        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
                indexes = torch.zeros(n_q, len(ref_layout), dtype=torch.long).numpy()
        mask = torch.zeros(n_q, len(ref_layout), dtype=torch.bool).numpy()
                                indexes[:] = n_q * timesteps
                for s, sequence_coords in enumerate(ref_layout):
            for coords in sequence_coords:
                if coords.t < timesteps:
                    indexes[coords.q, s] = coords.t + coords.q * timesteps
                    mask[coords.q, s] = 1
        indexes = torch.from_numpy(indexes).to(device)
        mask = torch.from_numpy(mask).to(device)
        return indexes, mask

    def build_pattern_sequence(self, z: torch.Tensor, special_token: int, keep_only_valid_steps: bool = False):
        B, K, T = z.shape
        indexes, mask = self._build_pattern_sequence_scatter_indexes(
            T, K, keep_only_valid_steps=keep_only_valid_steps, device=str(z.device)
        )
        z = z.view(B, -1)
                z = torch.cat([z, torch.zeros_like(z[:, :1]) + special_token], dim=1)
        values = z[:, indexes.view(-1)]
        values = values.view(B, K, indexes.shape[-1])
        return values, indexes, mask

    def _build_reverted_sequence_scatter_indexes(self, sequence_steps: int, n_q: int,
                                                 keep_only_valid_steps: bool = False,
                                                 is_model_output: bool = False,
                                                 device: tp.Union[torch.device, str] = 'cpu'):
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
                timesteps = self.timesteps
        assert n_q == self.n_q, f"invalid number of codebooks for the sequence and the pattern: {n_q} != {self.n_q}"
        assert sequence_steps <= len(ref_layout), \
            f"sequence to revert is longer than the defined pattern: {sequence_steps} > {len(ref_layout)}"

                if is_model_output and self.starts_with_special_token():
            ref_layout = ref_layout[1:]

                indexes = torch.zeros(n_q, timesteps, dtype=torch.long).numpy()
        mask = torch.zeros(n_q, timesteps, dtype=torch.bool).numpy()
                indexes[:] = n_q * sequence_steps
        for s, sequence_codes in enumerate(ref_layout):
            if s < sequence_steps:
                for code in sequence_codes:
                    if code.t < timesteps:
                        indexes[code.q, code.t] = s + code.q * sequence_steps
                        mask[code.q, code.t] = 1
        indexes = torch.from_numpy(indexes).to(device)
        mask = torch.from_numpy(mask).to(device)
        return indexes, mask

    def revert_pattern_sequence(self, s: torch.Tensor, special_token: int, keep_only_valid_steps: bool = False):
        B, K, S = s.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=False, device=str(s.device)
        )
        s = s.view(B, -1)
                s = torch.cat([s, torch.zeros_like(s[:, :1]) + special_token], dim=1)
        values = s[:, indexes.view(-1)]
        values = values.view(B, K, indexes.shape[-1])
        return values, indexes, mask

    def revert_pattern_logits(self, logits: torch.Tensor, special_token: float, keep_only_valid_steps: bool = False):
        B, card, K, S = logits.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=True, device=logits.device
        )
        logits = logits.reshape(B, card, -1)
                logits = torch.cat([logits, torch.zeros_like(logits[:, :, :1]) + special_token], dim=-1)          values = logits[:, :, indexes.view(-1)]
        values = values.view(B, card, K, indexes.shape[-1])
        return values, indexes, mask


class CodebooksPatternProvider(ABC):
    def __init__(self, n_q: int, cached: bool = True):
        assert n_q > 0
        self.n_q = n_q
        self.get_pattern = lru_cache(100)(self.get_pattern)  
    @abstractmethod
    def get_pattern(self, timesteps: int) -> Pattern:
        raise NotImplementedError()


class DelayedPatternProvider(CodebooksPatternProvider):
    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None,
                 flatten_first: int = 0, empty_initial: int = 0):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))
        self.delays = delays
        self.flatten_first = flatten_first
        self.empty_initial = empty_initial
        assert len(self.delays) == self.n_q
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        omit_special_token = self.empty_initial < 0
        out: PatternLayout = [] if omit_special_token else [[]]
        max_delay = max(self.delays)
        if self.empty_initial:
            out += [[] for _ in range(self.empty_initial)]
        if self.flatten_first:
            for t in range(min(timesteps, self.flatten_first)):
                for q in range(self.n_q):
                    out.append([LayoutCoord(t, q)])
        for t in range(self.flatten_first, timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= self.flatten_first:
                    v.append(LayoutCoord(t_for_q, q))
            out.append(v)
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class ParallelPatternProvider(DelayedPatternProvider):
    def __init__(self, n_q: int, empty_initial: int = 0):
        super().__init__(n_q, [0] * n_q, empty_initial=empty_initial)


class UnrolledPatternProvider(CodebooksPatternProvider):
    FlattenedCodebook = namedtuple('FlattenedCodebook', ['codebooks', 'delay'])

    def __init__(self, n_q: int, flattening: tp.Optional[tp.List[int]] = None,
                 delays: tp.Optional[tp.List[int]] = None):
        super().__init__(n_q)
        if flattening is None:
            flattening = list(range(n_q))
        if delays is None:
            delays = [0] * n_q
        assert len(flattening) == n_q
        assert len(delays) == n_q
        assert sorted(flattening) == flattening
        assert sorted(delays) == delays
        self._flattened_codebooks = self._build_flattened_codebooks(delays, flattening)
        self.max_delay = max(delays)

    def _build_flattened_codebooks(self, delays: tp.List[int], flattening: tp.List[int]):
        flattened_codebooks: dict = {}
        for q, (inner_step, delay) in enumerate(zip(flattening, delays)):
            if inner_step not in flattened_codebooks:
                flat_codebook = UnrolledPatternProvider.FlattenedCodebook(codebooks=[q], delay=delay)
            else:
                flat_codebook = flattened_codebooks[inner_step]
                assert flat_codebook.delay == delay, (
                    "Delay and flattening between codebooks is inconsistent: ",
                    "two codebooks flattened to the same position should have the same delay."
                )
                flat_codebook.codebooks.append(q)
            flattened_codebooks[inner_step] = flat_codebook
        return flattened_codebooks

    @property
    def _num_inner_steps(self):
        return max([inner_step for inner_step in self._flattened_codebooks.keys()]) + 1

    def num_virtual_steps(self, timesteps: int) -> int:
        return timesteps * self._num_inner_steps + 1

    def get_pattern(self, timesteps: int) -> Pattern:
                        indexed_out: list = [(-1, [])]
        max_timesteps = timesteps + self.max_delay
        for t in range(max_timesteps):
                                    for step in range(self._num_inner_steps):
                if step in self._flattened_codebooks:
                                        step_codebooks = self._flattened_codebooks[step]
                    t_for_q = t + step_codebooks.delay
                    coords = [LayoutCoord(t, q) for q in step_codebooks.codebooks]
                    if t_for_q < max_timesteps and t < max_timesteps:
                        indexed_out.append((t_for_q, coords))
                else:
                                        indexed_out.append((t, []))
        out = [coords for _, coords in sorted(indexed_out)]
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class CoarseFirstPattern(CodebooksPatternProvider):
    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None):
        super().__init__(n_q)
        if delays is None:
            delays = [0] * (n_q - 1)
        self.delays = delays
        assert len(self.delays) == self.n_q - 1
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        for t in range(timesteps):
            out.append([LayoutCoord(t, 0)])
        max_delay = max(self.delays)
        for t in range(timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= 0:
                    v.append(LayoutCoord(t_for_q, q + 1))
            out.append(v)
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class MusicLMPattern(CodebooksPatternProvider):
    def __init__(self, n_q: int, group_by: int = 2):
        super().__init__(n_q)
        self.group_by = group_by

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        for offset in range(0, self.n_q, self.group_by):
            for t in range(timesteps):
                for q in range(offset, offset + self.group_by):
                    out.append([LayoutCoord(t, q)])
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)
