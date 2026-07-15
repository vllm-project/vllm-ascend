from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from numbers import Integral
from typing import Any

from vllm.config import ParallelConfig


class BatchResultShapeError(RuntimeError):
    pass


def require_aligned_batch_results(
    operation: str,
    keys: list[str],
    results: Iterable[int] | None,
) -> list[int]:
    try:
        raw_values = list(results) if results is not None else []
    except (TypeError, ValueError) as exc:
        raise BatchResultShapeError(f"{operation} returned non-integer batch results") from exc
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw_values):
        raise BatchResultShapeError(f"{operation} returned non-integer batch results")
    values = [int(value) for value in raw_values]
    if len(values) != len(keys):
        raise BatchResultShapeError(f"{operation} returned {len(values)} results for {len(keys)} keys")
    return values


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

    def batch_add_lease(self, keys: list[str], lease_ttl_ms: int = 0) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_add_lease")

    def batch_remove_lease(self, keys: list[str]) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_remove_lease")

    def validate_layerwise_support(self) -> None:
        return None

    def batch_commit(self, keys: list[str]) -> list[int]:
        return [0] * len(keys)

    def batch_revoke(self, keys: list[str]) -> list[int]:
        return [0] * len(keys)

    def batch_put_start(self, keys: list[str], sizes: list[int]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_put_start")

    def batch_get_start(self, keys: list[str]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_get_start")

    def batch_get_end(self, keys: list[str]) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_get_end")

    def batch_copy_put(
        self,
        keys: list[str],
        all_buffers: list[list[int]],
        all_sizes: list[list[int]],
        all_dst_offsets: list[list[int]],
    ) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_copy_put")

    def batch_copy_get(
        self,
        keys: list[str],
        all_buffers: list[list[int]],
        all_sizes: list[list[int]],
        all_src_offsets: list[list[int]],
    ) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_copy_get")

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass
