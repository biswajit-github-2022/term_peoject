

import logging
import os
from pathlib import Path
import re
import typing as tp

import omegaconf

from .utils.cluster import _guess_cluster_type


logger = logging.getLogger(__name__)


class AudioCraftEnvironment:
    _instance = None
    DEFAULT_TEAM = "default"

    def __init__(self) -> None:
        cls._instance = None

    @classmethod
    def get_team(cls) -> str:
        return cls.instance().team

    @classmethod
    def get_cluster(cls) -> str:
        return cls.instance().cluster

    @classmethod
    def get_dora_dir(cls) -> Path:
        cluster_config = cls.instance()._get_cluster_config()
        dora_dir = os.getenv("AUDIOCRAFT_DORA_DIR", cluster_config["dora_dir"])
        logger.warning(f"Dora directory: {dora_dir}")
        return Path(dora_dir)

    @classmethod
    def get_reference_dir(cls) -> Path:
        cluster_config = cls.instance()._get_cluster_config()
        return Path(os.getenv("AUDIOCRAFT_REFERENCE_DIR", cluster_config["reference_dir"]))

    @classmethod
    def get_slurm_exclude(cls) -> tp.Optional[str]:

        Args:
            partition_types (list[str], optional): partition types to retrieve. Values must be
                from ['global', 'team']. If not provided, the global partition is returned.

        Args:
            path (str or Path): Path to resolve.
        Returns:
            Path: Resolved path.
        If no rules are defined, the path is returned as-is.
