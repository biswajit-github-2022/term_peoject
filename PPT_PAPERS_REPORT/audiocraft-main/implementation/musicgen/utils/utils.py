
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import wraps, lru_cache
import hashlib
import json
import logging
from pathlib import Path
import typing as tp

import flashy
import flashy.distrib
import omegaconf
import torch
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


def model_hash(model: torch.nn.Module) -> str:
    hasher = hashlib.sha1()
    for p in model.parameters():
        hasher.update(p.data.cpu().numpy().tobytes())
    return hasher.hexdigest()


def dict_from_config(cfg: omegaconf.DictConfig) -> dict:
    dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(dct, dict)
    return dct


def random_subset(dataset, max_samples: int, seed: int = 42) -> torch.utils.data.Subset:
    if max_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)
    return torch.utils.data.Subset(dataset, perm[:max_samples].tolist())


def get_loader(dataset, num_samples: tp.Optional[int], batch_size: int,
               num_workers: int, seed: int, **kwargs) -> torch.utils.data.DataLoader:
    if num_samples is not None:
        dataset = random_subset(dataset, num_samples, seed)

    dataloader = flashy.distrib.loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )
    return dataloader


def get_dataset_from_loader(dataloader):
    dataset = dataloader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset.dataset
    else:
        return dataset


def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers, mp_context=None):
        pass

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return


def get_pool_executor(num_workers: int, mp_context=None):
    return ProcessPoolExecutor(num_workers, mp_context) if num_workers > 1 else DummyPoolExecutor(1)


def length_to_mask(lengths: torch.Tensor, max_len: tp.Optional[int] = None) -> torch.Tensor:
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(final_length, 1)      return torch.arange(final_length, device=lengths.device)[None, :] < lengths[:, None]


def hash_trick(word: str, vocab_size: int) -> int:
    hash = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
    return hash % vocab_size


def with_rank_rng(base_seed: int = 1234):
    def _decorator(fun: tp.Callable):
        @wraps(fun)
        def _decorated(*args, **kwargs):
            state = torch.get_rng_state()
            seed = base_seed ^ flashy.distrib.rank()
            torch.manual_seed(seed)
            logger.debug('Rank dependent seed set to %d', seed)
            try:
                return fun(*args, **kwargs)
            finally:
                torch.set_rng_state(state)
                logger.debug('RNG state restored.')
        return _decorated
    return _decorator


def collate(tensors: tp.List[torch.Tensor], dim: int = 0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    tensors = [x.transpose(0, dim) for x in tensors]
    lens = torch.LongTensor([len(x) for x in tensors])
    padded_tensors = pad_sequence(tensors)
    padded_tensors = padded_tensors.transpose(0, 1)
    padded_tensors = padded_tensors.transpose(1, dim + 1)
    return padded_tensors, lens


def copy_state(state: tp.Any, device: tp.Union[torch.device, str] = 'cpu',
               dtype: tp.Optional[torch.dtype] = None) -> tp.Any:
    if isinstance(state, torch.Tensor):
        if dtype is None or not state.is_floating_point():
            dtype = state.dtype
        return state.detach().to(device=device, dtype=dtype, copy=True)
    elif isinstance(state, dict):
        return {k: copy_state(v, device, dtype) for k, v in state.items()}
    elif isinstance(state, list):
        return [copy_state(v, device, dtype) for v in state]


@contextmanager
def swap_state(model, state, **kwargs):
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state, **kwargs)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


@lru_cache(None)
def warn_once(logger, msg):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def load_clap_state_dict(clap_model, path: tp.Union[str, Path]):
    from clap_module.factory import load_state_dict      pkg = load_state_dict(path)
    pkg.pop('text_branch.embeddings.position_ids', None)
    clap_model.model.load_state_dict(pkg)
