"""Carry transfer records from Worker selection to Backend I/O.

The batch values retain dynamic Block IDs, keys, request
ownership, and current object selection together with the minimal bound facts
needed to interpret them: physical layers and remote object size. Reusable
layout formulas remain in the Infrastructure Mapping; session progress and
visibility fences remain in execution Session.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .rows import StoreTransferRecords

UInt64Array: TypeAlias = NDArray[np.uint64]
IndexArray: TypeAlias = NDArray[np.intp]
BoolArray: TypeAlias = NDArray[np.bool_]
KeysByPartition: TypeAlias = tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True, eq=False)
class KVGroupBatch:
    """One group's records: request ownership, content identity and local block ID.

    Each partition pairs with every record to form a transfer candidate. Candidate
    positions remain stable through selection, including repeated keys/block IDs.
    """

    group_id: int
    local_block_ids: UInt64Array
    keys_by_partition: KeysByPartition
    request_transfer_record_boundaries: IndexArray
    physical_layer_ids: tuple[int, ...]
    remote_object_size_bytes: int
    transfer_candidate_mask: BoolArray | None = None
    selected_key_values: tuple[str, ...] | None = None
    physical_layer_ids_by_partition: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.physical_layer_ids_by_partition is not None and len(self.physical_layer_ids_by_partition) != len(
            self.keys_by_partition
        ):
            raise ValueError("Per-partition physical layers must align keys_by_partition")

    @property
    def transfer_record_count(self) -> int:
        return len(self.local_block_ids)

    @property
    def transfer_candidate_count(self) -> int:
        return len(self.keys_by_partition) * self.transfer_record_count

    def selected_keys(self) -> tuple[str, ...]:
        if self.selected_key_values is not None:
            return self.selected_key_values
        if self.transfer_candidate_mask is None:
            return tuple(key for axis in self.keys_by_partition for key in axis)
        return tuple(
            key
            for partition_index, axis in enumerate(self.keys_by_partition)
            for transfer_record_index, key in enumerate(axis)
            if self.transfer_candidate_mask[partition_index * self.transfer_record_count + transfer_record_index]
        )

    def select_keys(self, accepted: set[str], claimed: set[str] | None = None) -> KVGroupBatch:
        current = self.transfer_candidate_mask
        selected: BoolArray = np.zeros(self.transfer_candidate_count, dtype=np.bool_)
        selected_keys: list[str] = []
        for partition_index, axis in enumerate(self.keys_by_partition):
            partition_candidate_start = partition_index * self.transfer_record_count
            for transfer_record_index, key in enumerate(axis):
                index = partition_candidate_start + transfer_record_index
                if (current is not None and not current[index]) or key not in accepted:
                    continue
                if claimed is not None and key in claimed:
                    continue
                selected[index] = True
                selected_keys.append(key)
                if claimed is not None:
                    claimed.add(key)
        selected.flags.writeable = False
        return replace(self, transfer_candidate_mask=selected, selected_key_values=tuple(selected_keys))

    def request_index(self, transfer_candidate_index: int) -> int:
        transfer_record_index = transfer_candidate_index % self.transfer_record_count
        return int(np.searchsorted(self.request_transfer_record_boundaries, transfer_record_index, side="right") - 1)

    def source(
        self, transfer_candidate_index: int, layer_id: int | None, *, request_index: int | None = None
    ) -> TransferSource:
        partition_index, transfer_record_index = divmod(transfer_candidate_index, self.transfer_record_count)
        layer_ids: tuple[int, ...]
        if layer_id is not None:
            layer_ids = (layer_id,)
        elif self.physical_layer_ids_by_partition is None:
            layer_ids = self.physical_layer_ids
        else:
            layer_ids = self.physical_layer_ids_by_partition[partition_index]
        if request_index is None:
            request_index = self.request_index(transfer_candidate_index)
        key = self.keys_by_partition[partition_index][transfer_record_index]
        return TransferSource(self, transfer_candidate_index, transfer_record_index, request_index, layer_ids, key)


@dataclass(frozen=True, slots=True)
class KVTransferBatch:
    """One Worker invocation with request ownership retained on group records."""

    request_ids: tuple[str, ...]
    groups: tuple[KVGroupBatch, ...]
    store_job_ids: tuple[int | None, ...] | None = None
    selected_key_values: tuple[str, ...] | None = None

    @property
    def empty(self) -> bool:
        return not self.selected_keys()

    def selected_keys(self) -> tuple[str, ...]:
        if self.selected_key_values is not None:
            return self.selected_key_values
        return tuple(key for group in self.groups for key in group.selected_keys())

    def select_keys(self, accepted: set[str], *, claim_once: bool = False) -> KVTransferBatch:
        claimed: set[str] | None = set() if claim_once else None
        groups = tuple(group.select_keys(accepted, claimed) for group in self.groups)
        selected_keys = tuple(key for group in groups for key in group.selected_keys())
        return replace(self, groups=groups, selected_key_values=selected_keys)

    def for_layer(self, layer_id: int) -> KVTransferBatch:
        groups = tuple(group for group in self.groups if layer_id in group.physical_layer_ids)
        selected_keys = tuple(key for group in groups for key in group.selected_keys())
        return replace(self, groups=groups, selected_key_values=selected_keys)


@dataclass(frozen=True, slots=True)
class TransferSource:
    """Immutable request and cache provenance for one Backend object result."""

    group: KVGroupBatch
    transfer_candidate_index: int
    transfer_record_index: int
    request_index: int
    physical_layer_ids: tuple[int, ...]
    key: str

    @property
    def group_id(self) -> int:
        return self.group.group_id

    @property
    def local_block_id(self) -> int:
        return int(self.group.local_block_ids[self.transfer_record_index])


@dataclass(frozen=True, slots=True)
class StoreGroupTransferCandidates:
    """One group's list-backed transfer records and keys by partition."""

    group_id: int
    transfer_records_by_request: tuple[StoreTransferRecords, ...]
    keys_by_partition: KeysByPartition

    @property
    def transfer_record_count(self) -> int:
        return sum(len(rows[0]) for rows in self.transfer_records_by_request)

    @property
    def transfer_candidate_count(self) -> int:
        return len(self.keys_by_partition) * self.transfer_record_count


@dataclass(frozen=True, slots=True)
class StoreTransferCandidates:
    """Pre-admission Store candidates in group/partition/transfer-record order."""

    groups: tuple[StoreGroupTransferCandidates, ...]


def store_transfer_candidate_keys(candidates: StoreTransferCandidates) -> tuple[str, ...]:
    """Flatten candidate keys in group/partition/transfer-record order."""

    return tuple(key for group in candidates.groups for axis in group.keys_by_partition for key in axis)


def select_store_transfer_candidates(
    candidate_keys: tuple[str, ...], accepted: set[str], *, claim_once: bool
) -> tuple[bool, ...]:
    """Map key admission back to ordered candidate objects."""

    if not claim_once:
        return tuple(key in accepted for key in candidate_keys)
    claimed = set()
    selected = []
    for key in candidate_keys:
        include = key in accepted and key not in claimed
        selected.append(include)
        if include:
            claimed.add(key)
    return tuple(selected)
