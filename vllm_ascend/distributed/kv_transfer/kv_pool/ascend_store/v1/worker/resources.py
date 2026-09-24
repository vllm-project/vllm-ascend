"""Worker-owned cache resources shared by classic KV Pool operations."""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase, KeyMetadata
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


def _physical_layer_index(layer_name: str) -> int:
    layer = re.search(r"layers\.(\d+)", layer_name)
    if layer is not None:
        return int(layer.group(1))
    first_number = re.search(r"\d+", layer_name)
    return int(first_number.group()) if first_number is not None else 0


class ClassicMooncakeBackend(Backend):
    """Expose classic Mooncake I/O while owning its registered buffers."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._registered_buffer_addresses: tuple[int, ...] = ()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self.requires_exists_before_put = backend.requires_exists_before_put

    def set_device(self) -> None:
        self._backend.set_device()

    def register_buffer(self, ptrs: list[int], lengths: list[int]) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Classic Mooncake backend is closed")
            self._backend.register_buffer(ptrs, lengths)
            self._registered_buffer_addresses += tuple(ptrs)

    def exists(self, keys: list[str]) -> list[int]:
        return self._backend.exists(keys)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        return self._backend.put(keys, addrs, sizes)

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        return self._backend.get(keys, addrs, sizes)

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._unregister_buffers()
            self._close_store()
            self._closed = True

    def _unregister_buffers(self) -> None:
        if not self._registered_buffer_addresses:
            return
        with global_te.register_buffer_lock:
            if global_te.transfer_engine is None:
                raise RuntimeError("Mooncake transfer engine is unavailable while unregistering KV buffers")
            result = global_te.transfer_engine.batch_unregister_memory(list(self._registered_buffer_addresses))
            if result != 0:
                raise RuntimeError(f"Mooncake memory unregistration failed with result: {result}")
            global_te.is_register_buffer = False
        self._registered_buffer_addresses = ()

    def _close_store(self) -> None:
        store = self._backend.store
        if store is None:
            return
        result = store.close()
        if result != 0:
            raise RuntimeError(f"Mooncake store close failed with result: {result}")
        self._backend.store = None


class WorkerCacheResources:
    """Own the backend, token database and registered Worker KV tensors."""

    def __init__(self, backend: ClassicMooncakeBackend, token_database: ChunkedTokenDatabase, num_blocks: int) -> None:
        self.backend = backend
        self.token_database = token_database
        self.num_blocks = num_blocks
        self.kv_caches: dict[str, torch.Tensor] | None = None

    @classmethod
    def create(
        cls,
        parallel_config: ParallelConfig,
        extra_config: dict[str, Any],
        key_metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int,
        num_blocks: int,
    ) -> WorkerCacheResources:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

        token_database = ChunkedTokenDatabase([key_metadata], [block_size], None, hash_block_size)
        backend = ClassicMooncakeBackend(MooncakeBackend(parallel_config, extra_config=extra_config))
        return cls(backend, token_database, num_blocks)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.kv_caches = kv_caches
        self._register_kv_buffers(kv_caches)

    def close(self) -> None:
        self.backend.close()
        self.kv_caches = None

    def _register_kv_buffers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        group_addresses: list[int] = []
        group_block_lengths: list[int] = []
        group_block_strides: list[int] = []
        registered_regions: dict[int, tuple[int, int]] = {}

        for layer_name in sorted(kv_caches, key=lambda name: (_physical_layer_index(name), name)):
            cache_or_caches = kv_caches[layer_name]
            caches = (cache_or_caches,) if isinstance(cache_or_caches, torch.Tensor) else tuple(cache_or_caches)
            for cache in caches:
                assert cache.shape[0] % self.num_blocks == 0, (
                    "The external block size must be an integer multiple of the kernel block size."
                )
                block_scale = cache.shape[0] // self.num_blocks
                block_length = cache[0].numel() * cache.element_size() * block_scale
                block_stride = cache.stride(0) * cache.element_size() * block_scale
                address = cache.data_ptr()
                region_end = address + (self.num_blocks - 1) * block_stride + block_length
                storage_key = cache.untyped_storage().data_ptr()
                previous = registered_regions.get(storage_key)
                registered_regions[storage_key] = (
                    (min(previous[0], address), max(previous[1], region_end))
                    if previous is not None
                    else (address, region_end)
                )
                group_addresses.append(address)
                group_block_lengths.append(block_length)
                group_block_strides.append(block_stride)

        self.token_database.set_group_buffers({0: group_addresses}, {0: group_block_lengths}, {0: group_block_strides})
        self.backend.register_buffer(
            [start for start, _ in registered_regions.values()],
            [end - start for start, end in registered_regions.values()],
        )
