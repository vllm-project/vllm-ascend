"""Scheduler-to-Worker KV transfer commands and their vLLM envelope."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_utils import BlockHash

from ..graph.coordinates import TokenRange


@dataclass(frozen=True, slots=True)
class LoadCommand:
    """A Scheduler-approved Load command."""

    request_id: str
    load_range: TokenRange
    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[BlockHash, ...]


@dataclass(frozen=True, slots=True)
class StoreCommand:
    """A Scheduler-approved asynchronous Store command."""

    request_id: str
    store_range: TokenRange
    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[BlockHash, ...]
    num_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class LoadCommandBatch:
    """Load commands approved for one Worker step."""

    commands: tuple[LoadCommand, ...] = ()


@dataclass(frozen=True, slots=True)
class StoreCommandBatch:
    """Store commands approved for one Worker step."""

    commands: tuple[StoreCommand, ...] = ()


@dataclass(frozen=True, slots=True)
class KVTransferStep(KVConnectorMetadata):
    """One Scheduler-to-Worker step carrying approved transfer commands."""

    load: LoadCommandBatch = LoadCommandBatch()
    store: StoreCommandBatch = StoreCommandBatch()
