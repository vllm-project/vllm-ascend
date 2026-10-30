"""Submission ownership, completion and transfer facts shared by services and transports."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ...backend import GVARegion
from ...infrastructure.mapping import PartitionId
from ...protocol.messages import StoreCommand
from ...protocol.results import LoadResult, StoreResult
from .selection import CachePartition, LoadSelection, StoreSelection


@dataclass(slots=True)
class LayerTransferPlan:
    # Layer -> participating partitions; no GVA layer Key lists are materialized.
    layers: dict[int, tuple[tuple[int, PartitionId], ...]] = field(default_factory=dict)
    regions: dict[str, GVARegion] = field(default_factory=dict)
    block_ids: dict[tuple[int, PartitionId], NDArray[np.uint64]] = field(default_factory=dict)
    object_bases: dict[tuple[int, PartitionId], NDArray[np.uint64] | NDArray[np.object_]] = field(default_factory=dict)
    # Slots in batch.codes: local destinations for write, unique objects for read.
    result_slots: dict[CachePartition, tuple[int, ...]] = field(default_factory=dict)
    # Key-range consumes this route on every layer result; GVA derives it only on failure.
    range_indices: dict[int, tuple[int, ...]] = field(default_factory=dict)
    range_keys: dict[int, list[str]] = field(default_factory=dict)
    key_merges: dict[int, tuple[int, ...] | None] = field(default_factory=dict)

    def result_indices(self, layer_id: int) -> Iterator[int]:
        if layer_id in self.range_indices:
            yield from self.range_indices[layer_id]
        else:
            for partition in self.layers[layer_id]:
                yield from self.result_slots[partition]

    def failures(self, layer_id: int, codes: int | tuple[int, ...]) -> Iterator[tuple[int, int]]:
        if layer_id not in self.range_keys:
            if not isinstance(codes, int):
                raise ValueError("Backend GVA result must be one integer")
            if codes != 0:
                for index in self.result_indices(layer_id):
                    yield index, codes
            return
        if isinstance(codes, int) or len(codes) != len(self.range_keys[layer_id]):
            raise ValueError("Backend result count does not match layer parameters")
        merges = self.key_merges[layer_id]
        results = codes if merges is None else (codes[position] for position in merges)
        for index, code in zip(self.range_indices[layer_id], results, strict=True):
            if code != 0:
                yield index, code


@dataclass(slots=True)
class LoadBatch:
    """One submission's ownership, final business completion and optional per-layer progress."""

    request_ids: tuple[str, ...]
    selection: LoadSelection
    completion: Future[LoadResult] = field(init=False)
    destination_writes_stopped: bool = True
    plan: LayerTransferPlan | None = None
    # One result per local destination, including all targets of a shared object.
    codes: list[int] = field(default_factory=list)
    copied_layers: set[int] = field(default_factory=set)
    error: BaseException | None = None


@dataclass(slots=True, eq=False)
class StoreBatch:
    """One submission's source ownership, final completion and optional per-layer read progress."""

    commands: tuple[StoreCommand, ...]
    needs_shared_confirmation: bool = False
    request_ids: frozenset[str] = field(init=False)
    job_ids: frozenset[int] = field(init=False)
    completion: Future[tuple[StoreResult, ...]] = field(init=False)
    reported_jobs: set[int] = field(default_factory=set)
    selection: StoreSelection | None = None
    identities: Mapping[str, object] = field(default_factory=dict)
    open_failed: bool = False
    plan: LayerTransferPlan | None = None
    # Compact execution-object slots; exists hits never enter this batch.
    codes: list[int | None] = field(default_factory=list)
    opened_indices: set[int] = field(default_factory=set)
    remaining_objects: int = 0
    unsafe_objects: set[int] = field(default_factory=set)
    error: Exception | None = None
    contract_error: Exception | None = None
    fatal_error: Exception | None = None

    def __post_init__(self) -> None:
        self.request_ids = frozenset(command.request_id for command in self.commands)
        self.job_ids = frozenset(command.store_job_id for command in self.commands if command.store_job_id is not None)
