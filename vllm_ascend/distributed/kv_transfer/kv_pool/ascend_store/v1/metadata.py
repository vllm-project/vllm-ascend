"""Cross-role execution requests used by the classic path."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True, slots=True)
class LoadRequest:
    """A Scheduler-approved synchronous Load command."""

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


class AscendStoreV1Metadata(KVConnectorMetadata):
    def __init__(self, preempted_req_ids: set[str]) -> None:
        self.load_requests: list[LoadRequest] = []
        self.store_requests: list[StoreRequest] = []
        self.preempted_req_ids = preempted_req_ids
