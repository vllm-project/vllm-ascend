"""Own Backend state, key derivation and registered local KV memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ..backend import BackendAdapter, create_backend
from ..graph.elements import KVMemoryGeometry, KVMemorySegment

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

    from ..graph.topology import KVPoolGroupTopology


class KVPoolResources:
    """Own the Backend, token database and registered local KV tensors."""

    def __init__(
        self,
        backend: BackendAdapter,
        token_database: ChunkedTokenDatabase,
        num_blocks: int,
        groups: tuple[KVPoolGroupTopology, ...],
    ) -> None:
        self.backend = backend
        self.token_database = token_database
        self.num_blocks = num_blocks
        self._groups = groups
        self.kv_caches: dict[str, torch.Tensor] | None = None
        self._registered = False
        self._closed = False

    @classmethod
    def create(
        cls,
        parallel_config: ParallelConfig,
        extra_config: dict[str, Any],
        groups: tuple[KVPoolGroupTopology, ...],
        hash_block_size: int,
        num_blocks: int,
        consumer_pipeline_partitions: tuple[int, ...] | None,
    ) -> KVPoolResources:
        backend_name = extra_config.get("backend", "mooncake").strip().lower()
        token_database = ChunkedTokenDatabase(
            [group.key_metadata for group in groups],
            [group.block_size for group in groups],
            None if consumer_pipeline_partitions is None else list(consumer_pipeline_partitions),
            hash_block_size,
        )
        backend = create_backend(backend_name, parallel_config, extra_config)
        return cls(backend, token_database, num_blocks, groups)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> KVMemoryGeometry:
        if self._closed:
            raise RuntimeError("KV resources are closed")
        if self._registered:
            raise RuntimeError("KV caches are already registered")
        memory_geometry = self._register_kv_buffers(kv_caches)
        self.kv_caches = kv_caches
        self._registered = True
        return memory_geometry

    def close(self) -> None:
        if self._closed:
            return
        close_backend = getattr(self.backend, "close", None)
        if callable(close_backend):
            close_backend()
        self.kv_caches = None
        self._closed = True

    def _register_kv_buffers(self, kv_caches: dict[str, torch.Tensor]) -> KVMemoryGeometry:
        group_ids = tuple(group.group_id for group in self._groups)
        addresses_by_group: dict[int, list[int]] = {group_id: [] for group_id in group_ids}
        block_lengths_by_group: dict[int, list[int]] = {group_id: [] for group_id in group_ids}
        block_strides_by_group: dict[int, list[int]] = {group_id: [] for group_id in group_ids}
        segments_by_group: dict[int, list[KVMemorySegment]] = {group_id: [] for group_id in group_ids}
        registered_regions: dict[int, tuple[int, int]] = {}

        for group in self._groups:
            for layer in group.layers:
                for layer_name in layer.layer_names:
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
                        segments_by_group[group.group_id].append(
                            KVMemorySegment(
                                layer_name,
                                layer.physical_layer_id,
                                address,
                                block_length,
                                block_stride,
                                block_length // group.block_size,
                            )
                        )
                        addresses_by_group[group.group_id].append(address)
                        block_lengths_by_group[group.group_id].append(block_length)
                        block_strides_by_group[group.group_id].append(block_stride)
                        region_end = address + (self.num_blocks - 1) * block_stride + block_length
                        storage_key = cache.untyped_storage().data_ptr()
                        previous = registered_regions.get(storage_key)
                        registered_regions[storage_key] = (
                            (min(previous[0], address), max(previous[1], region_end))
                            if previous is not None
                            else (address, region_end)
                        )

        self.token_database.set_group_buffers(
            addresses_by_group,
            block_lengths_by_group,
            block_strides_by_group,
            group_num_layers={group.group_id: len(group.layer_names) for group in self._groups},
        )
        self.backend.register_buffer(
            [start for start, _ in registered_regions.values()],
            [end - start for start, end in registered_regions.values()],
        )
        return {group_id: tuple(segments) for group_id, segments in segments_by_group.items()}
