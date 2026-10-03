"""Register Worker KV memory and expose the facts consumed by rule binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from ..backend import BackendSpec

if TYPE_CHECKING:
    from ..program.spec.topology import KVPoolGroupTopology

_GVA_ALIGNMENT = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class GVAObjectLayout:
    """Global coordinates shared by every GVA object produced by this rank."""

    pipeline_layer_start: int
    total_hidden_layers: int
    dcp_rank: int
    dcp_size: int
    put_step: int


class KVPoolResources:
    """Own Backend state and extract registered memory facts once."""

    def __init__(
        self,
        backend: Backend,
        backend_spec: BackendSpec,
        num_blocks: int,
        groups: tuple[KVPoolGroupTopology, ...],
        *,
        gva_layout: GVAObjectLayout | None = None,
        align_shared_storage: bool = False,
    ) -> None:
        self.backend = backend
        self.backend_spec = backend_spec
        self.num_blocks = num_blocks
        self._groups = groups
        self._gva_layout = gva_layout
        self._align_shared_storage = align_shared_storage
        self.kv_caches: dict[str, torch.Tensor] | None = None
        self._memory_bound = False
        self._closed = False

    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("KV resources are closed")
        if self._memory_bound:
            raise RuntimeError("KV caches are already bound")
        registration = self._register_kv_buffers(kv_caches)
        self.kv_caches = kv_caches
        self._memory_bound = True
        return registration

    def close(self) -> None:
        if self._closed:
            return
        close_backend = getattr(self.backend, "close", None)
        if callable(close_backend):
            close_backend()
        self.kv_caches = None
        self._closed = True

    def _register_kv_buffers(self, kv_caches: dict[str, torch.Tensor]) -> dict[str, Any]:
        base_addresses: dict[int, list[int]] = {}
        block_lengths: dict[int, list[int]] = {}
        block_strides: dict[int, list[int]] = {}
        layer_entry_offsets: dict[int, list[int]] = {}
        registered_regions: dict[int, tuple[int, int]] = {}

        for group in self._groups:
            bases: list[int] = []
            lengths: list[int] = []
            strides: list[int] = []
            layer_offsets = [0]
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
                        bases.append(address)
                        lengths.append(block_length)
                        strides.append(block_stride)
                        region_end = address + (self.num_blocks - 1) * block_stride + block_length
                        storage_key = cache.untyped_storage().data_ptr()
                        previous = registered_regions.get(storage_key)
                        registered_regions[storage_key] = (
                            (min(previous[0], address), max(previous[1], region_end))
                            if previous is not None
                            else (address, region_end)
                        )
                layer_offsets.append(len(bases))
            base_addresses[group.group_id] = bases
            block_lengths[group.group_id] = lengths
            block_strides[group.group_id] = strides
            layer_entry_offsets[group.group_id] = layer_offsets

        if self._align_shared_storage:
            for storage_key, (start, end) in registered_regions.items():
                aligned_start = start // _GVA_ALIGNMENT * _GVA_ALIGNMENT
                if aligned_start < storage_key:
                    raise ValueError("Shared KV storage cannot satisfy the GVA alignment boundary")
                registered_regions[storage_key] = aligned_start, end
        self.backend.register_buffer(
            [start for start, _ in registered_regions.values()],
            [end - start for start, end in registered_regions.values()],
        )

        registration: dict[str, Any] = {
            "base_addresses": base_addresses,
            "block_lengths": block_lengths,
            "block_strides": block_strides,
            "layer_entry_offsets": layer_entry_offsets,
        }
        if self._gva_layout is not None:
            object_sizes, object_offsets = self._resolve_gva_objects(block_lengths)
            registration["object_sizes"] = object_sizes
            registration["object_offsets"] = object_offsets
        return registration

    def _resolve_gva_objects(
        self,
        block_lengths: dict[int, list[int]],
    ) -> tuple[dict[int, int], dict[int, int]]:
        layout = self._gva_layout
        assert layout is not None
        object_sizes: dict[int, int] = {}
        object_offsets: dict[int, int] = {}
        for group in self._groups:
            local_size = sum(block_lengths[group.group_id])
            local_layer_count = len(group.layers)
            if local_layer_count <= 0:
                raise ValueError(f"GVA cache group {group.group_id} has no local layer")
            bytes_per_layer = (local_size + local_layer_count - 1) // local_layer_count
            pipeline_offset = layout.pipeline_layer_start * bytes_per_layer
            global_layer_slots = max(
                layout.total_hidden_layers,
                layout.pipeline_layer_start + local_layer_count,
            )
            global_span = max(global_layer_slots * bytes_per_layer, pipeline_offset + local_size)
            if layout.dcp_size > 1 and layout.put_step > 1:
                shard_stride = _align_up(global_span, _GVA_ALIGNMENT)
                object_offsets[group.group_id] = pipeline_offset + layout.dcp_rank * shard_stride
                object_sizes[group.group_id] = shard_stride * layout.dcp_size
            else:
                object_offsets[group.group_id] = pipeline_offset
                object_sizes[group.group_id] = global_span
        return object_sizes, object_offsets


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
