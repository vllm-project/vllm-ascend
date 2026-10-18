"""Generate one layer's transfers into the same complete objects as Bulk.

    Binding: once                         Batch: each prepare() call
    BulkMapping                           keys + block_ids + mask
         | LayerwiseMapping(...)                  | prepare(...)
         v                                        v
    cached layer_layouts                  LayerwiseSelection
         |                                        |
         +-------------------+--------------------+
                             v
          +-----------------------------------------+
          | arguments(selection, layer_id=...)      |
          |   choose layer_layouts[layer_id]        |
          |   join by key_prefix_index              |
          +-----------------------------------------+
                             |
                             v
              keys, local addresses, lengths, object offsets

Object offsets are relative to the complete Bulk object, never rebased per layer
or call order. Local layouts share Bulk's read-only arrays; readiness, allocation
and transfer execution remain caller-owned.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .bulk import BulkMapping
from .types import LayerTransferArguments, LocalMemorySegments, UInt64Array


@dataclass(frozen=True, slots=True)
class LayerTransferLayout:
    """Fixed memory layout for one layer and key prefix, built with LayerwiseMapping.

    Example: one prefix, two layers, one segment per layer. Addresses, strides,
    lengths and object offsets below are in bytes.

        complete object:  0 [--- L0: 32 ---] 32 [-- L1: 16 --] 48
        L1 layout:        object_offsets_bytes=[32], object_size_bytes=48
        local L1 segment: base=2000, stride=128, length=16
                              |
                              +-- runtime block ID=3 --> address=2384

    Base, stride and length come from local_memory_segments; actual addresses
    require a Selection's block IDs. No concrete keys/IDs are stored in the layout.
    The enclosing layer_layouts dictionary supplies layer ID; key_prefix_index
    joins this layout to Selection rows. Object size includes ALL assigned layers.
    """

    key_prefix_index: int
    local_memory_segments: LocalMemorySegments
    object_offsets_bytes: UInt64Array
    object_size_bytes: int


@dataclass(frozen=True, slots=True)
class _PrefixSelection:
    """Each position pairs a complete object key with its local cache block ID."""

    keys: tuple[str, ...]
    block_ids: UInt64Array


@dataclass(frozen=True, slots=True)
class LayerwiseSelection:
    """One prepare() call's selected object keys and local cache block IDs.

    Example with simplified key prefixes P0: and P1: in bulk.key_prefixes order:

        by_key_prefix[0]     object key       local block ID
          row 0              P0:A <---------> 3
          row 1              P0:B <---------> 7
        by_key_prefix[1] = None               no selected rows

    This selects block rows, NOT layers: reuse it across layer_id calls without
    filtering again. It snapshots keys/IDs, not transfer addresses. selected_row_count
    totals rows across prefixes (2 above), including duplicates. bulk requires the
    exact same Bulk instance when consuming the selection.
    """

    bulk: BulkMapping
    by_key_prefix: tuple[_PrefixSelection | None, ...]
    selected_row_count: int


class LayerwiseMapping:
    """Reuse Bulk layouts and cache the local-to-object byte positions for each layer."""

    def __init__(self, bulk: BulkMapping) -> None:
        self.bulk = bulk
        layers: dict[int, list[LayerTransferLayout]] = defaultdict(list)
        segment_offsets_bytes_by_layout: dict[tuple[int, int], UInt64Array] = {}
        for key_prefix_index, (layer_ids, boundaries, segments, object_size) in enumerate(
            zip(
                bulk.physical_layer_ids_by_key_prefix,
                bulk.layer_segment_boundaries_by_key_prefix,
                bulk.local_memory_segments_by_key_prefix,
                bulk.object_sizes_bytes_by_key_prefix,
                strict=True,
            )
        ):
            segment_count = len(segments.block_zero_addresses)
            if not segment_count:
                continue
            # Head slices share lengths; never scan the same length array per head.
            layout_key = (id(segments.segment_lengths_bytes), segment_count)
            offsets = segment_offsets_bytes_by_layout.get(layout_key)
            if offsets is None:
                offsets = np.empty(segment_count + 1, dtype=np.uint64)
                offsets[0] = 0
                np.cumsum(np.broadcast_to(segments.segment_lengths_bytes, (segment_count,)), out=offsets[1:])
                offsets.flags.writeable = False
                segment_offsets_bytes_by_layout[layout_key] = offsets
            if int(offsets[-1]) != object_size:
                raise ValueError("Bulk segments must cover the complete object size")
            for layer_index, layer_id in enumerate(layer_ids):
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
                    LayerTransferLayout(key_prefix_index, layer_segments, offsets[start:end], object_size)
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
        """Snapshot rows for reuse across layers; mask order is key prefix, then input row.

        Keys come from bulk.keys(). Selection is caller-owned, including
        admission and duplicate writers. Repeated input rows remain repeated here.
        No local address matrix or per-layer Backend list is allocated yet.
        """
        ids = np.asarray(block_ids, dtype=np.uint64)
        if ids.ndim != 1:
            raise ValueError("Block IDs must be one-dimensional rows")
        row_count = len(ids)
        key_prefix_count = len(self.bulk.key_prefixes)
        if len(keys) != key_prefix_count or any(len(prefix_keys) != row_count for prefix_keys in keys):
            raise ValueError("Keys must align local key prefixes and Block ID rows")
        selections = None
        if selected_keys is not None:
            selections = np.asarray(selected_keys, dtype=np.bool_)
            if selections.ndim != 1 or selections.size != key_prefix_count * row_count:
                raise ValueError("Key selection must be a flat key-prefix-major row mask")
            selections = selections.reshape(key_prefix_count, row_count)
        if not row_count:
            return LayerwiseSelection(self.bulk, (None,) * key_prefix_count, 0)

        selections_by_prefix: list[_PrefixSelection | None] = []
        shared_ids = None
        selected_row_count = 0
        for key_prefix_index, prefix_keys in enumerate(keys):
            selected_rows = None
            if selections is not None:
                selected_count = np.count_nonzero(selections[key_prefix_index])
                if not selected_count:
                    selections_by_prefix.append(None)
                    continue
                if selected_count != row_count:
                    selected_rows = np.flatnonzero(selections[key_prefix_index])
            if selected_rows is None:
                if shared_ids is None:
                    shared_ids = ids.copy()
                    shared_ids.flags.writeable = False
                selected_block_ids = shared_ids
                selected_prefix_keys = tuple(prefix_keys)
            else:
                selected_block_ids = ids[selected_rows]
                selected_block_ids.flags.writeable = False
                selected_prefix_keys = tuple(prefix_keys[row] for row in selected_rows)
            selections_by_prefix.append(_PrefixSelection(selected_prefix_keys, selected_block_ids))
            selected_row_count += len(selected_block_ids)
        return LayerwiseSelection(self.bulk, tuple(selections_by_prefix), selected_row_count)

    def arguments(self, selection: LayerwiseSelection, *, layer_id: int) -> LayerTransferArguments:
        """Combine selected keys/block IDs with this layer's cached transfer layouts.

        The selection must belong to the same Bulk mapping instance.
        Absent or empty layers and unselected objects produce no arguments.
        Each returned Python row is independently mutable; cached layouts and
        prepared selections remain unchanged across Load/Store and repeated calls.
        """
        if selection.bulk is not self.bulk:
            raise ValueError("Prepared selections belong to a different Bulk mapping")
        layer = self.layer_layouts.get(layer_id)
        if not selection.selected_row_count or layer is None:
            return [], [], [], []
        keys, addresses, lengths, offsets = [], [], [], []
        for layout in layer:
            selected = selection.by_key_prefix[layout.key_prefix_index]
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
