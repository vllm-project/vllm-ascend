"""Operation-owned requests carried across the classic role boundary."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True, slots=True)
class LoadRequest:
    """A Scheduler-approved Load command."""

    request_id: str
    transfer_end_token: int
    block_ids: tuple[int, ...]
    block_hashes: tuple[BlockHash, ...]
    vllm_cached_tokens: int
    kvpool_cached_tokens: int


@dataclass(frozen=True, slots=True)
class StoreRequest:
    """A Scheduler-approved asynchronous Store command."""

    request_id: str
    save_end_token: int
    block_ids: tuple[int, ...]
    block_hashes: tuple[BlockHash, ...]
    num_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class LoadRequestBatch:
    """Load requests approved for one Worker step."""

    requests: tuple[LoadRequest, ...] = ()


@dataclass(frozen=True, slots=True)
class StoreRequestBatch:
    """Store requests and preempted work to discard in one Worker step."""

    requests: tuple[StoreRequest, ...] = ()
    preempted_request_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class AscendStoreV1Metadata(KVConnectorMetadata):
    load: LoadRequestBatch = LoadRequestBatch()
    store: StoreRequestBatch = StoreRequestBatch()
