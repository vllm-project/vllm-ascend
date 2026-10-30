"""Bind operation-specific ownership once, during selection."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TypeAlias

from ...infrastructure.mapping import PartitionId

CachePartition: TypeAlias = tuple[int, PartitionId]  # Cache group and its physical partition.


@dataclass(slots=True)
class LoadTargets:
    """Destination ownership in one physical partition; result slots follow this order."""

    object_indices: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    request_indices: list[int] = field(default_factory=list)
    start: int = 0


@dataclass(slots=True)
class LoadSelection:
    """Unique remote identities and every local destination, without reverse indices."""

    partition_order: dict[CachePartition, int]
    keys: list[str] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    local: dict[CachePartition, LoadTargets] = field(default_factory=dict)
    target_count: int = 0

    def add_target(self, index: int, partition: CachePartition, block_id: int, request_index: int, size: int) -> None:
        if index == len(self.sizes):
            self.sizes.append(size)
        elif self.sizes[index] != size:
            raise ValueError(f"Key {self.keys[index]!r} has inconsistent object sizes")
        local = self.local.get(partition)
        if local is None:
            local = self.local[partition] = LoadTargets()
        local.object_indices.append(index)
        local.block_ids.append(block_id)
        local.request_indices.append(request_index)

    def iter_local(self) -> Iterator[tuple[CachePartition, LoadTargets]]:
        for partition in self.partition_order:
            if partition in self.local:
                yield partition, self.local[partition]

    def bind_results(self) -> None:
        for _, local in self.iter_local():
            local.start = self.target_count
            self.target_count += len(local.block_ids)


# Physical partition, local Block ID and owning command index, bound during selection.
StoreSource: TypeAlias = tuple[CachePartition, int, int]


@dataclass(slots=True)
class StoreSelection:
    """Parallel unique Keys and sources; after exists, only missing objects remain."""

    keys: list[str] = field(default_factory=list)
    sources: list[StoreSource] = field(default_factory=list)
