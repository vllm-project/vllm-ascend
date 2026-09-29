"""Bind Backend state and registered local KV memory to one compiled program."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from ..backend import BackendSpec, create_backend, resolve_backend_spec
from ..program.representation import KVMemoryGeometry, KVMemorySegment

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

    from ..program.spec.topology import KVPoolGroupTopology


class KVPoolResources:
    """Own the Backend and local KV memory bound to one compiled program."""

    def __init__(
        self,
        backend: Backend,
        backend_spec: BackendSpec,
        num_blocks: int,
        groups: tuple[KVPoolGroupTopology, ...],
    ) -> None:
        self.backend = backend
        self.backend_spec = backend_spec
        self.num_blocks = num_blocks
        self._groups = groups
        self.kv_caches: dict[str, torch.Tensor] | None = None
        self._memory_bound = False
        self._closed = False

    @classmethod
    def bind(
        cls,
        backend_name: str,
        parallel_config: ParallelConfig,
        extra_config: dict[str, Any],
        groups: tuple[KVPoolGroupTopology, ...],
        num_blocks: int,
    ) -> KVPoolResources:
        backend_spec = resolve_backend_spec(backend_name)
        backend = create_backend(backend_spec, parallel_config, extra_config)
        return cls(backend, backend_spec, num_blocks, groups)

    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> KVMemoryGeometry:
        if self._closed:
            raise RuntimeError("KV resources are closed")
        if self._memory_bound:
            raise RuntimeError("KV caches are already bound")
        memory_geometry = self._register_kv_buffers(kv_caches)
        self.kv_caches = kv_caches
        self._memory_bound = True
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
                        region_end = address + (self.num_blocks - 1) * block_stride + block_length
                        storage_key = cache.untyped_storage().data_ptr()
                        previous = registered_regions.get(storage_key)
                        registered_regions[storage_key] = (
                            (min(previous[0], address), max(previous[1], region_end))
                            if previous is not None
                            else (address, region_end)
                        )

        self.backend.register_buffer(
            [start for start, _ in registered_regions.values()],
            [end - start for start, end in registered_regions.values()],
        )
        return {group_id: tuple(segments) for group_id, segments in segments_by_group.items()}
