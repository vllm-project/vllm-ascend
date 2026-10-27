"""Bind local sources and naming for complete-object transfers.

Bulk concatenates each object's assigned segments without holes. It therefore
needs payload lengths, but does not compile per-segment remote offsets or any
Layerwise state. Readiness, writer selection and Backend calls remain external.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .local_layout import bind_local_layout
from .object_naming import KeyBuilder, KeyNamer, bind_canonical_keys

if TYPE_CHECKING:
    import torch

    from ..configuration.topology import KVPoolTopology


class BulkMapping:
    """Bind independent Key naming and local byte geometry for one cache group."""

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
