"""Canonical PoolKey names; local prefixes share the globally bound Lookup strings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..types import KeysByObject

if TYPE_CHECKING:
    from ...configuration.topology import KVPoolTopology
    from ..local_layout import ObjectCoordinate


@dataclass(frozen=True, slots=True)
class CanonicalKeys:
    key_prefixes: tuple[str, ...]
    lookup_key_prefixes: tuple[str, ...]

    def keys(self, hashes: Sequence[bytes | str]) -> KeysByObject:
        return self._map_keys(self.key_prefixes, hashes)

    def lookup_keys(self, hashes: Sequence[bytes | str]) -> KeysByObject:
        return self._map_keys(self.lookup_key_prefixes, hashes)

    @staticmethod
    def _map_keys(prefixes: tuple[str, ...], hashes: Sequence[bytes | str]) -> KeysByObject:
        hash_strings = tuple(value if isinstance(value, str) else value.hex() for value in hashes)
        return tuple(tuple(prefix + value for value in hash_strings) for prefix in prefixes)


def bind_canonical_keys(
    topology: KVPoolTopology, group_id: int, coordinates: Sequence[ObjectCoordinate]
) -> CanonicalKeys:
    # The canonical codec imports Worker dependencies; defer it until binding.
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey

    group = next(group for group in topology.transfer_groups if group.group_id == group_id)
    partitions = topology.consumer_pipeline_partitions or ()
    pp_size = len(partitions) if len(partitions) > 1 else topology.pp_size
    head_count = topology.tp_size if group.uses_align_state else topology.tp_partition.key_rank_count
    lookup_prefixes = tuple(
        PoolKey(replace(group.key_metadata, pp_rank=pp, dcp_rank=dcp, head_or_tp_rank=head), "").to_string()
        for pp in range(pp_size)
        for dcp in range(topology.dcp_size)
        for head in range(head_count)
    )
    local_prefixes = tuple(
        lookup_prefixes[
            (coordinate.pp_rank * topology.dcp_size + coordinate.dcp_rank) * head_count + coordinate.head_rank
        ]
        for coordinate in coordinates
    )
    return CanonicalKeys(local_prefixes, lookup_prefixes)
