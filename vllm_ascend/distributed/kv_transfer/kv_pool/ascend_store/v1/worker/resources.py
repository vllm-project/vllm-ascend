"""Register Worker KV memory and expose the facts consumed by projection binding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..backend import (
    BackendSpec,
    BufferRegistrationError,
    KVStoreBackend,
    Registration,
)

if TYPE_CHECKING:
    from ..topology import KVPoolGroupTopology

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
        backend: KVStoreBackend,
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
        self.kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]] | None = None
        self._buffer_registration: Registration | None = None
        self._binding_started = False
        self._closed = False

    def bind_kv_caches(self, kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("KV resources are closed")
        if self._binding_started:
            raise RuntimeError("KV caches are already bound")
        self._binding_started = True
        self.kv_caches = kv_caches
        try:
            buffer_registration, registration = self._register_kv_buffers(kv_caches)
        except BufferRegistrationError as error:
            self._buffer_registration = error.registration
            raise
        self._buffer_registration = buffer_registration
        return registration

    def close(self) -> None:
        if self._closed:
            return
        if self._buffer_registration is not None:
            self._buffer_registration.close()
        self.backend.close()
        self.kv_caches = None
        self._closed = True

    def _register_kv_buffers(
        self,
        kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
    ) -> tuple[Registration, dict[str, Any]]:
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
                        # NoPE MLA exposes an empty RoPE view whose data_ptr() is zero.
                        if not cache.numel():
                            continue
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
                        # Payload lengths exclude padding; registration must still cover the last kernel block.
                        element_size = cache.element_size()
                        kernel_block_span = element_size + sum(
                            (size - 1) * stride * element_size
                            for size, stride in zip(cache.shape[1:], cache.stride()[1:], strict=True)
                        )
                        region_end = (
                            address
                            + (self.num_blocks * block_scale - 1) * cache.stride(0) * element_size
                            + kernel_block_span
                        )
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
        buffer_registration = self.backend.register_buffer(
            [start for start, _ in registered_regions.values()],
            [end - start for start, end in registered_regions.values()],
        )
        if not callable(getattr(buffer_registration, "close", None)):
            raise TypeError(f"{type(self.backend).__name__}.register_buffer must return a Registration owner")
        return buffer_registration, registration

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
