
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from functools import partial
from hashlib import sha1
import logging
from pathlib import Path
import sys
import typing as tp
import zipfile

import flashy
import torch


logger = logging.getLogger(__name__)


def get_full_embed(full_embed: torch.Tensor, x: tp.Any, idx: int, device: tp.Union[str, torch.device]) -> torch.Tensor:
    return full_embed.to(device)


class EmbeddingCache:
    def __init__(self, cache_path: tp.Union[str, Path], device: tp.Union[str, torch.device],
                 compute_embed_fn: tp.Callable[[Path, tp.Any, int], torch.Tensor],
                 extract_embed_fn: tp.Optional[tp.Callable[[torch.Tensor, tp.Any, int], torch.Tensor]] = None):
        self.cache_path = Path(cache_path)
        self.device = device
        self._compute_embed_fn = compute_embed_fn
        self._extract_embed_fn: tp.Callable[[torch.Tensor, tp.Any, int], torch.Tensor]
        if extract_embed_fn is not None:
            self._extract_embed_fn = extract_embed_fn
        else:
            self._extract_embed_fn = partial(get_full_embed, device=device)
        if self.cache_path is not None:
            self.cache_path.mkdir(exist_ok=True, parents=True)
            logger.info(f"Cache instantiated at: {self.cache_path}")
            self.pool = ThreadPoolExecutor(8)
            self.pool.__enter__()
        self._current_batch_cache: dict = {}
        self._memory_cache: dict = {}

    def _get_cache_path(self, path: tp.Union[Path, str]):
        try:
            embed = torch.load(cache, 'cpu')
        except Exception as exc:
            logger.error("Error loading %s: %r", cache, exc)
            embed = None
        return embed

    def get_embed_from_cache(self, paths: tp.List[Path], x: tp.Any) -> torch.Tensor:
        embeds = []
        for idx, path in enumerate(paths):
            cache = self._get_cache_path(path)
            if cache in self._current_batch_cache:
                embed = self._current_batch_cache[cache]
            else:
                full_embed = self._compute_embed_fn(path, x, idx)
                try:
                    with flashy.utils.write_and_rename(cache, pid=True) as f:
                        torch.save(full_embed.cpu(), f)
                except Exception as exc:
                    logger.error('Error saving embed %s (%s): %r', cache, full_embed.shape, exc)
                else:
                    logger.info('New embed cache saved: %s (%s)', cache, full_embed.shape)
                    embed = self._extract_embed_fn(full_embed, x, idx)
            embeds.append(embed)
        embed = torch.stack(embeds, dim=0)
        return embed

    def populate_embed_cache(self, paths: tp.List[Path], x: tp.Any) -> None:
        self._current_batch_cache.clear()
        if self.cache_path is not None:
            futures: list = []
            for path in paths:
                assert path is not None, "Path is required for computation from cache"
                cache = self._get_cache_path(path)
                if cache in self._memory_cache or not cache.exists():
                    futures.append(None)
                else:
                    futures.append(self.pool.submit(EmbeddingCache._get_full_embed_from_cache, cache))
            for idx, (path, future) in enumerate(zip(paths, futures)):
                assert path is not None
                cache = self._get_cache_path(path)
                full_embed = None
                if future is None:
                    if cache in self._memory_cache:
                        full_embed = self._memory_cache[cache]
                else:
                    full_embed = future.result()
                    if full_embed is not None:
                        self._memory_cache[cache] = full_embed
                        full_embed = full_embed.to(self.device)
                if full_embed is not None:
                    embed = self._extract_embed_fn(full_embed, x, idx)
                    self._current_batch_cache[cache] = embed


class CachedBatchWriter:
    def __init__(self, cache_folder: Path):
        self.cache_folder = cache_folder
        self._current_epoch: tp.Optional[int] = None
        self._current_index = 0

    def start_epoch(self, epoch: int):
        self._current_epoch = epoch
        self._current_index = 0
        self._zip_path.parent.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _get_zip_path(cache_folder: Path, epoch: int, index: int):
        return cache_folder / f"{epoch:05d}" / f"{index:06d}.zip"

    @property
    def _zip_path(self):
        assert self._current_epoch is not None
        return CachedBatchWriter._get_zip_path(self.cache_folder, self._current_epoch, self._current_index)

    def save(self, *content):
        all_contents = []
        for rank in range(flashy.distrib.world_size()):
            their_content = flashy.distrib.broadcast_object(content, src=rank)
            all_contents.append(their_content)

        if flashy.distrib.is_rank_zero():
            idx = 0
            with flashy.utils.write_and_rename(self._zip_path) as tmp:
                with zipfile.ZipFile(tmp, 'w') as zf:
                    for content in all_contents:
                        for vals in zip(*content):
                            with zf.open(f'{idx}', 'w') as f:                                  torch.save(vals, f)
                            idx += 1
        flashy.distrib.barrier()
        self._current_index += 1


class CachedBatchLoader:

    def __init__(self, cache_folder: Path, batch_size: int,
                 num_workers: int = 10, min_length: int = 1):
        self.cache_folder = cache_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_length = min_length
        self._current_epoch: tp.Optional[int] = None
        self.sampler = None  
    def __len__(self):
        path = CachedBatchWriter._get_zip_path(self.cache_folder, self._current_epoch or 0, 0).parent
        return len([p for p in path.iterdir() if p.suffix == ".zip"])

    def start_epoch(self, epoch: int):
        self._current_epoch = epoch

    def _zip_path(self, index: int):
        assert self._current_epoch is not None
        return CachedBatchWriter._get_zip_path(self.cache_folder, self._current_epoch, index)

    def _load_one(self, index: int):
        zip_path = self._zip_path(index)
        if not zip_path.exists():
            if index < self.min_length:
                raise RuntimeError(f"Cache should have at least {self.min_length} batches, but {index} doesn't exist")

            return None
        mode = "rb" if sys.version_info >= (3, 9) else "r"
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                rank = flashy.distrib.rank()
                world_size = flashy.distrib.world_size()
                root = zipfile.Path(zf)
                items = list(root.iterdir())
                total_batch_size = self.batch_size * world_size
                if len(items) < total_batch_size:
                    raise RuntimeError(
                        f"The cache can handle a max batch size of {len(items)}, "
                        f"but {total_batch_size} is needed.")
                start = rank * self.batch_size
                items = items[start: start + self.batch_size]
                assert len(items) == self.batch_size
                entries = []
                entries = [torch.load(item.open(mode), 'cpu') for item in items]                  transposed = zip(*entries)
                out = []
                for part in transposed:
                    assert len(part) > 0
                    if isinstance(part[0], torch.Tensor):
                        out.append(torch.stack(part))
                    else:
                        assert isinstance(part, torch.Tensor)
                        out.append(part)
                return out
        except Exception:
            logger.error("Error when reading zip path %s", zip_path)
            raise

    def __iter__(self):
        pool = ThreadPoolExecutor(self.num_workers)
        next_index = 0
        queue = deque()

        def _get_next():
            nonlocal next_index
            r = queue.popleft().result()
            if r is None:
                return None
            else:
                queue.append(pool.submit(self._load_one, next_index))
                next_index += 1
            return r

        with pool:
                        for _ in range(2 * self.num_workers):
                queue.append(pool.submit(self._load_one, next_index))
                next_index += 1
            while True:
                batch = _get_next()
                if batch is None:
                    return
                yield batch
