from abc import ABC, abstractmethod
from typing import Any

from vllm.config import ParallelConfig


class Backend(ABC):
    @abstractmethod
    def __init__(self, parallel_config: ParallelConfig, **kwargs: Any):
        """Backends accept the parallel config plus a ``**kwargs`` bag for
        connector-level overrides (e.g. ``kv_connector_extra_config``).
        Backends that don't care can simply ignore the extras."""
        pass

    @abstractmethod
    def set_device(self):
        pass

    @abstractmethod
    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        pass

    @abstractmethod
    def exists(self, keys: list[str]) -> list[int]:
        pass

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass
