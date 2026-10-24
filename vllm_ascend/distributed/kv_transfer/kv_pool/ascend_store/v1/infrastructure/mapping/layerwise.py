"""Generate one layer's transfers from bound local segments and object geometry.

    Binding: once                         Batch: each prepare() call
    registration + topology               keys + block_ids + mask
         | LayerwiseMapping(...)                  | prepare(...)
         v                                        v
    cached layer_layouts                  LayerwiseSelection
         |                                        |
         +-------------------+--------------------+
                             v
          +-----------------------------------------+
          | arguments(selection, layer_id=...)      |
          |   choose layer_layouts[layer_id]        |
          |   join by object_index                  |
          +-----------------------------------------+
                             |
                             v
              keys, local addresses, lengths, object offsets

The selected object-layout function runs once. Layer arrays share this Mapping's
local sources, not a separately constructed Bulk view. Readiness, remote
allocation and transfer execution remain caller-owned.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .local_layout import LocalMemorySegments, bind_local_layout
from .object_layout import ObjectLayoutFactory, build_default_object_layouts
from .object_naming import KeyBuilder, KeyNamer, bind_canonical_keys
from .types import LayerTransferArguments, UInt64Array

if TYPE_CHECKING:
    import torch

    from ..configuration.topology import KVPoolTopology


@dataclass(frozen=True, slots=True)
class LayerTransferLayout:
    """Fixed memory layout for one layer and local object, built with LayerwiseMapping.

    Example: one object, two layers, one segment per layer. Addresses, strides,
    lengths and object offsets below are in bytes.

        complete object:  0 [--- L0: 32 ---] 32 [-- L1: 16 --] 48
        L1 layout:        object_offsets_bytes=[32], object_size_bytes=48
        local L1 segment: base=2000, stride=128, length=16
                              |
                              +-- runtime block ID=3 --> address=2384

    Base, stride and length come from local_memory_segments; actual addresses
    require a Selection's block IDs. No concrete keys/IDs are stored in the layout.
    The enclosing layer_layouts dictionary supplies layer ID; object_index
    joins this layout to Selection rows. Object size includes ALL assigned layers.
    """

    object_index: int
    local_memory_segments: LocalMemorySegments
    object_offsets_bytes: UInt64Array
    object_size_bytes: int


@dataclass(frozen=True, slots=True)
class _ObjectSelection:
    """Selected keys/IDs and their original input rows; no request ownership."""

    keys: tuple[str, ...]
    block_ids: UInt64Array
    row_indices: range | NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class LayerwiseSelection:
    """One prepare() call's selected object keys and local cache block IDs.

    Example with simplified names P0: and P1: in bound local object order:

        by_object[0]     object key       local block ID
          row 0             P0:A <---------> 3
          row 1             P0:B <---------> 7
        by_object[1] = None                no selected rows

    This selects block rows, NOT layers: reuse it across layer_id calls without
    filtering again. It snapshots keys/IDs and input-row positions, not transfer
    addresses or request ownership. selected_row_count totals rows across objects
    (2 above), including duplicates. The caller keeps these object axes aligned
    with the group's bound layer layouts and resolves request ownership separately.
    """

    by_object: tuple[_ObjectSelection | None, ...]
    selected_row_count: int


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
        object_layout_factory: ObjectLayoutFactory | None = None,
    ) -> None:
        # ========================================
        # Bind local sources and wire identity
        # ========================================

        self.local = bind_local_layout(
            topology,
            entry_addresses_by_group,
            entry_block_payload_bytes_by_group,
            entry_block_strides_bytes_by_group,
            layer_entry_boundaries_by_group,
            group_id=group_id,
            kv_cache_layout=kv_cache_layout,
            kv_caches=kv_caches,
        )
        coordinates = tuple(source.coordinate for source in self.local.objects)
        self.naming: KeyNamer = (
            bind_canonical_keys(topology, self.local.group_id, coordinates)
            if key_builder is None
            else key_builder.bind(
                self.local.group_id, coordinates, lookup_head_count=topology.tp_partition.key_rank_count
            )
        )
        bind_objects = build_default_object_layouts if object_layout_factory is None else object_layout_factory
        self.object_layouts = bind_objects(self.local)

        # ========================================
        # Compile layer access over bound objects
        # ========================================
        #
        # Slice the existing arrays; this does not rebuild local geometry or
        # change the source/object order selected by the binding functions.

        layers: dict[int, list[LayerTransferLayout]] = defaultdict(list)
        for object_index, (local_object, remote_object) in enumerate(
            zip(self.local.objects, self.object_layouts, strict=True)
        ):
            segments = local_object.memory_segments
            boundaries = local_object.layer_segment_boundaries
            for layer_index, layer_id in enumerate(local_object.physical_layer_ids):
                start, end = boundaries[layer_index], boundaries[layer_index + 1]
                if start == end:
                    continue
                strides, lengths = segments.group_block_strides_bytes, segments.segment_lengths_bytes
                layer_segments = LocalMemorySegments(
                    segments.block_zero_addresses[start:end],
                    strides if len(strides) == 1 else strides[start:end],
                    lengths if len(lengths) == 1 else lengths[start:end],
                )
                layers[layer_id].append(
                    LayerTransferLayout(
                        object_index,
                        layer_segments,
                        remote_object.segment_offsets_bytes[start:end],
                        remote_object.size_bytes,
                    )
                )
        self.layer_layouts: Mapping[int, tuple[LayerTransferLayout, ...]] = MappingProxyType(
            {layer_id: tuple(layouts) for layer_id, layouts in layers.items()}
        )

    def prepare(
        self,
        keys: Sequence[Sequence[str]],
        block_ids: Sequence[int] | UInt64Array,
        *,
        selected_keys: Sequence[bool] | None = None,
    ) -> LayerwiseSelection:
        """Snapshot rows for reuse across layers; mask order is local object, then input row.

        Keys come from this mapping's bound identity. Selection is caller-owned, including
        admission and duplicate writers. Repeated input rows remain repeated here.
        No local address matrix or per-layer Backend list is allocated yet.
        """
        ids = np.asarray(block_ids, dtype=np.uint64)
        if ids.ndim != 1:
            raise ValueError("Block IDs must be one-dimensional rows")
        row_count = len(ids)
        object_count = len(self.local.objects)
        if len(keys) != object_count or any(len(object_keys) != row_count for object_keys in keys):
            raise ValueError("Keys must align local objects and Block ID rows")
        selections = None
        if selected_keys is not None:
            selections = np.asarray(selected_keys, dtype=np.bool_)
            if selections.ndim != 1 or selections.size != object_count * row_count:
                raise ValueError("Key selection must be a flat object-major row mask")
            selections = selections.reshape(object_count, row_count)
        if not row_count:
            return LayerwiseSelection((None,) * object_count, 0)

        selections_by_object: list[_ObjectSelection | None] = []
        shared_ids = None
        selected_row_count = 0
        for object_index, object_keys in enumerate(keys):
            selected_rows = None
            if selections is not None:
                selected_count = np.count_nonzero(selections[object_index])
                if not selected_count:
                    selections_by_object.append(None)
                    continue
                if selected_count != row_count:
                    selected_rows = np.flatnonzero(selections[object_index])
            if selected_rows is None:
                if shared_ids is None:
                    shared_ids = ids.copy()
                    shared_ids.flags.writeable = False
                selected_block_ids = shared_ids
                selected_object_keys = tuple(object_keys)
            else:
                selected_block_ids = ids[selected_rows]
                selected_block_ids.flags.writeable = False
                selected_object_keys = tuple(object_keys[row] for row in selected_rows)
                selected_rows.flags.writeable = False
            row_indices = range(row_count) if selected_rows is None else selected_rows
            selections_by_object.append(_ObjectSelection(selected_object_keys, selected_block_ids, row_indices))
            selected_row_count += len(selected_block_ids)
        return LayerwiseSelection(tuple(selections_by_object), selected_row_count)

    def arguments(self, selection: LayerwiseSelection, *, layer_id: int) -> LayerTransferArguments:
        """Combine selected keys/block IDs with this layer's cached transfer layouts.

        The caller supplies the selection prepared for this group's key axes.
        Absent or empty layers and unselected objects produce no arguments.
        Each returned Python row is independently mutable; cached layouts and
        prepared selections remain unchanged across Load/Store and repeated calls.
        """
        layer = self.layer_layouts.get(layer_id)
        if not selection.selected_row_count or layer is None:
            return [], [], [], []
        keys: list[str] = []
        addresses: list[list[int]] = []
        lengths: list[list[int]] = []
        offsets: list[list[int]] = []
        for layout in layer:
            selected = selection.by_object[layout.object_index]
            if selected is None:
                continue
            segments = layout.local_memory_segments
            local = selected.block_ids[:, None] * segments.group_block_strides_bytes + segments.block_zero_addresses
            sizes = np.broadcast_to(segments.segment_lengths_bytes, segments.block_zero_addresses.shape).tolist()
            positions = layout.object_offsets_bytes.tolist()
            keys.extend(selected.keys)
            addresses.extend(local.tolist())
            lengths.extend(sizes.copy() for _ in selected.keys)
            offsets.extend(positions.copy() for _ in selected.keys)
        return keys, addresses, lengths, offsets
