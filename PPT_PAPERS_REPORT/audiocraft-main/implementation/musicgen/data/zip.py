
import typing
import zipfile

from dataclasses import dataclass
from functools import lru_cache
from typing_extensions import Literal


DEFAULT_SIZE = 32
MODE = Literal['r', 'w', 'x', 'a']


@dataclass(order=True)
class PathInZip:

    INFO_PATH_SEP = ':'
    zip_path: str
    file_path: str

    def __init__(self, path: str) -> None:
        split_path = path.split(self.INFO_PATH_SEP)
        assert len(split_path) == 2
        self.zip_path, self.file_path = split_path

    @classmethod
    def from_paths(cls, zip_path: str, file_path: str):
        return cls(zip_path + cls.INFO_PATH_SEP + file_path)

    def __str__(self) -> str:
        return self.zip_path + self.INFO_PATH_SEP + self.file_path


def _open_zip(path: str, mode: MODE = 'r'):
    return zipfile.ZipFile(path, mode)


_cached_open_zip = lru_cache(DEFAULT_SIZE)(_open_zip)


def set_zip_cache_size(max_size: int):
    global _cached_open_zip
    _cached_open_zip = lru_cache(max_size)(_open_zip)


def open_file_in_zip(path_in_zip: PathInZip, mode: str = 'r') -> typing.IO:
    zf = _cached_open_zip(path_in_zip.zip_path)
    return zf.open(path_in_zip.file_path)
