from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vllm.config import ParallelConfig


class Backend(ABC):
    store: Any | None = None

    @abstractmethod
    def __init__(self, parallel_config: ParallelConfig):
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

    def batch_get_key_info(self, keys: list[str]):
        raise NotImplementedError(f"{type(self).__name__} does not support batch_get_key_info")

    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_alloc")

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass
