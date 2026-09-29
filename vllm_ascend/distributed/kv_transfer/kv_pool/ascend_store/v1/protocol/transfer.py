"""KV transfer messages exchanged across the Scheduler-Worker boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata, KVConnectorWorkerMetadata
from vllm.v1.core.kv_cache_utils import BlockHash

from ..coordinates import TokenRange
from .lookup import TailKeyBoundary


@dataclass(frozen=True, slots=True)
class LoadCommand:
    """A Scheduler-approved Load command."""

    request_id: str
    load_range: TokenRange
    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[BlockHash, ...]
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


@dataclass(frozen=True, slots=True)
class RangeStoreCommand:
    """Store KV selected from one Scheduler-approved token range."""

    request_id: str
    store_range: TokenRange
    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[BlockHash, ...]
    num_prompt_tokens: int
    store_job_id: int | None = None


@dataclass(frozen=True, slots=True)
class StateCheckpointSource:
    """One exact Mamba state checkpoint handed off by vLLM."""

    group_id: int
    block_id: int
    boundary_token: int


@dataclass(frozen=True, slots=True)
class CheckpointStoreCommand:
    """Store exact state checkpoints without resolving a mutable block table."""

    request_id: str
    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[BlockHash, ...]
    published_store_end_token: int
    sources: tuple[StateCheckpointSource, ...]
    store_job_id: int | None = None


StoreCommand: TypeAlias = RangeStoreCommand | CheckpointStoreCommand


@dataclass(frozen=True, slots=True)
class LoadCommandBatch:
    """Load commands approved for one Worker step."""

    commands: tuple[LoadCommand, ...] = ()


@dataclass(frozen=True, slots=True)
class StoreCommandBatch:
    """Store commands grouped by whether their source is already ready."""

    source_pending_commands: tuple[StoreCommand, ...] = ()
    source_ready_commands: tuple[StoreCommand, ...] = ()

    @property
    def commands(self) -> tuple[StoreCommand, ...]:
        return self.source_pending_commands + self.source_ready_commands

    @property
    def all_sources_ready(self) -> bool:
        return bool(self.source_ready_commands) and not self.source_pending_commands


@dataclass(frozen=True, slots=True)
class KVTransferStep(KVConnectorMetadata):
    """One Scheduler-to-Worker step carrying approved transfer commands."""

    load: LoadCommandBatch = LoadCommandBatch()
    store: StoreCommandBatch = StoreCommandBatch()


@dataclass(slots=True)
class StoreSourceReleaseMetadata(KVConnectorWorkerMetadata):
    """Count Worker ranks that have stopped reading each Store job's source."""

    released_store_jobs: dict[int, int] = field(default_factory=dict)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> StoreSourceReleaseMetadata:
        if not isinstance(other, StoreSourceReleaseMetadata):
            raise TypeError(f"Cannot aggregate {type(other).__name__} into StoreSourceReleaseMetadata")
        for store_job_id, count in other.released_store_jobs.items():
            self.released_store_jobs[store_job_id] = self.released_store_jobs.get(store_job_id, 0) + count
        return self
