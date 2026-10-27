"""Bind reusable per-layer local geometry and remote object offsets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from .local_layout import LocalMemorySegments, bind_local_layout
from .object_layout import RemoteObjectLayoutFactory, build_default_object_layouts
from .object_naming import KeyBuilder, KeyNamer, bind_canonical_keys
from .types import UInt64Array

if TYPE_CHECKING:
    import torch

    from ..configuration.topology import KVPoolTopology


@dataclass(frozen=True, slots=True)
class LayerPartitionLayout:
    """Fixed memory layout for one layer and local object, built with LayerwiseMapping.

    Example: one object, two layers, one segment per layer. Addresses, strides,
    lengths and object offsets below are in bytes.

        complete object:  0 [--- L0: 32 ---] 32 [-- L1: 16 --] 48
        L1 layout:        remote_segment_offsets_bytes=[32], remote_object_size_bytes=48
        local L1 segment: base=2000, stride=128, length=16
                              |
                              +-- runtime block ID=3 --> address=2384

    Base, stride and length come from local_segments; actual addresses
    require the Worker's selected block IDs. No concrete keys/IDs are stored in the layout.
    The enclosing layer_layouts dictionary supplies layer ID; partition_index
    joins this layout to the Worker's partition selection. Object size includes ALL assigned layers.
    """

    partition_index: int
    local_segments: LocalMemorySegments
    remote_segment_offsets_bytes: UInt64Array
    remote_object_size_bytes: int


class LayerwiseMapping:
    """Expose bound local/naming capabilities and cache per-layer object access layouts."""

    def __init__(
        self,
        topology: KVPoolTopology,
        entry_addresses_by_group: Mapping[int, Sequence[int]],
        entry_block_payload_bytes_by_group: Mapping[int, Sequence[int]],
        entry_block_strides_bytes_by_group: Mapping[int, Sequence[int]],
        layer_entry_boundaries_by_group: Mapping[int, Sequence[int]],
        *,
        group_id: int | None = None,
        kv_cache_layout: str = "NHD",
        kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]] | None = None,
        key_builder: KeyBuilder | None = None,
        remote_object_layout_factory: RemoteObjectLayoutFactory | None = None,
    ) -> None:
        # ========================================
        # Bind local sources and wire identity
        # ========================================

        self.local_layout = bind_local_layout(
            topology,
            entry_addresses_by_group,
            entry_block_payload_bytes_by_group,
            entry_block_strides_bytes_by_group,
            layer_entry_boundaries_by_group,
            group_id=group_id,
            kv_cache_layout=kv_cache_layout,
            kv_caches=kv_caches,
        )
        coordinates = tuple(source.coordinate for source in self.local_layout.partitions)
        self.key_namer: KeyNamer = (
            bind_canonical_keys(topology, self.local_layout.group_id, coordinates)
            if key_builder is None
            else key_builder.bind(
                self.local_layout.group_id, coordinates, lookup_head_count=topology.tp_partition.key_rank_count
            )
        )
        bind_objects = (
            build_default_object_layouts if remote_object_layout_factory is None else remote_object_layout_factory
        )
        self.remote_object_layouts = bind_objects(self.local_layout)

        # ========================================
        # Compile layer access over bound objects
        # ========================================
        #
        # Slice the existing arrays; this does not rebuild local geometry or
        # change the source/object order selected by the binding functions.

        layers: dict[int, list[LayerPartitionLayout]] = defaultdict(list)
        for partition_index, (local_object, remote_object) in enumerate(
            zip(self.local_layout.partitions, self.remote_object_layouts, strict=True)
        ):
            segments = local_object.local_segments
            boundaries = local_object.layer_segment_index_boundaries
            for layer_index, layer_id in enumerate(local_object.physical_layer_ids):
                start, end = boundaries[layer_index], boundaries[layer_index + 1]
                if start == end:
                    continue
                strides, lengths = segments.group_block_strides_bytes, segments.segment_lengths_bytes
                layer_segments = LocalMemorySegments(
                    segments.block_zero_segment_addresses[start:end],
                    strides if len(strides) == 1 else strides[start:end],
                    lengths if len(lengths) == 1 else lengths[start:end],
                )
                layers[layer_id].append(
                    LayerPartitionLayout(
                        partition_index,
                        layer_segments,
                        remote_object.segment_boundary_offsets_bytes[start:end],
                        remote_object.size_bytes,
                    )
                )
        self.layer_layouts: Mapping[int, tuple[LayerPartitionLayout, ...]] = MappingProxyType(
            {layer_id: tuple(layouts) for layer_id, layouts in layers.items()}
        )
