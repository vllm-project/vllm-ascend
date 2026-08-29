from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vllm.config import ParallelConfig


class GVALayerwiseCapable(ABC):
    """Optional protocol family for backends that support the layerwise GVA
    transfer mode (batch key-info lookup, allocation, and leases).

    The generic layers must gate these calls behind
    ``backend_supports(backend_name, "gva_layerwise")`` instead of calling
    them unconditionally on a plain :class:`Backend`.
    """

    @abstractmethod
    def batch_get_key_info(self, keys: list[str]) -> list[Any]:
        pass

    @abstractmethod
    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        pass

    @abstractmethod
    def batch_add_lease(self, keys: list[str], lease_ttl_ms: int = 0) -> list[int]:
        pass

    @abstractmethod
    def batch_remove_lease(self, keys: list[str]) -> int:
        pass

    @abstractmethod
    def batch_write_finish(self, keys: list[str], results: list[int]) -> list[int]:
        pass


class Backend(ABC):
    store: Any | None = None

    @abstractmethod
    def __init__(self, parallel_config: ParallelConfig, lazy_init: bool = False):
        pass

    @classmethod
    def create_scheduler_client(cls, parallel_config: ParallelConfig):
        return cls(parallel_config)

    @abstractmethod
    def set_device(self):
        pass

    @abstractmethod
    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        pass

    @abstractmethod
    def exists(self, keys: list[str]) -> list[int]:
        pass

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return self.exists(keys)

    def on_worker_ready(self) -> None:  # noqa: B027 (optional lifecycle hook)
        """Called after kv caches are registered and before transfer threads
        start. Backends that need eager initialization override this.
        """

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass
