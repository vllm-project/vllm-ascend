"""Bind local sources and naming for complete-object transfers.

Bulk concatenates each object's assigned segments without holes. It therefore
needs payload lengths, but does not compile per-segment remote offsets or any
Layerwise state. Readiness, writer selection and Backend calls remain external.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .local_layout import bind_local_layout
from .object_naming import KeyBuilder, KeyNamer, bind_canonical_keys
from .types import BulkTransferArguments, KeysByObject, UInt64Array

if TYPE_CHECKING:
    import torch

    from ..configuration.topology import KVPoolTopology


class BulkMapping:
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
    ) -> None:
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

    def keys(self, hashes: Sequence[bytes | str]) -> KeysByObject:
        return self.naming.keys(hashes)

    def lookup_keys(self, hashes: Sequence[bytes | str]) -> KeysByObject:
        return self.naming.lookup_keys(hashes)

    def arguments(
        self,
        keys: Sequence[Sequence[str]],
        block_ids: Sequence[int] | UInt64Array,
        *,
        selected_keys: Sequence[bool] | None = None,
    ) -> BulkTransferArguments:
        """Select complete payloads without clipping to logical token coverage.

        Input keys retain every row, including unselected rows. The optional
        flat mask is object-major, then input-row-major. Repeated IDs and input
        order are preserved. Callers ensure allocation bounds and object validity.
        """
        ids = np.asarray(block_ids, dtype=np.uint64)
        if ids.ndim != 1:
            raise ValueError("Block IDs must be one-dimensional rows")
        row_count = len(ids)
        objects = self.local.objects
        if len(keys) != len(objects) or any(len(object_keys) != row_count for object_keys in keys):
            raise ValueError("Keys must align local objects and Block ID rows")
        selections = None
        if selected_keys is not None:
            selections = np.asarray(selected_keys, dtype=np.bool_)
            if selections.ndim != 1 or selections.size != len(keys) * row_count:
                raise ValueError("Key selection must be a flat object-major row mask")
            selections = selections.reshape(len(keys), row_count)

        backend_keys: list[str] = []
        backend_addresses: list[list[int]] = []
        backend_lengths: list[list[int]] = []
        if not row_count:
            return backend_keys, backend_addresses, backend_lengths

        for object_index, (object_keys, obj) in enumerate(zip(keys, objects, strict=True)):
            segments = obj.memory_segments
            selected_rows = None
            if selections is not None:
                selected_count = np.count_nonzero(selections[object_index])
                if not selected_count:
                    continue
                if selected_count != row_count:
                    selected_rows = np.flatnonzero(selections[object_index])
            selected_block_ids = ids if selected_rows is None else ids[selected_rows]
            backend_keys.extend(object_keys if selected_rows is None else (object_keys[row] for row in selected_rows))
            addresses = selected_block_ids[:, None] * segments.group_block_strides_bytes + segments.block_zero_addresses
            lengths = np.broadcast_to(segments.segment_lengths_bytes, segments.block_zero_addresses.shape).tolist()
            backend_addresses.extend(addresses.tolist())
            # Lists belong to individual Backend rows, even when their lengths match.
            backend_lengths.extend(lengths.copy() for _ in selected_block_ids)
        return backend_keys, backend_addresses, backend_lengths
