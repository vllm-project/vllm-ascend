"""Scheduler-owned vLLM objects projected into the KVCacheServer process."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHash

if TYPE_CHECKING:
    from ..metadata import AscendStoreKVConnectorWorkerMetadata


@dataclass
class RequestView:
    """Snapshot of a vLLM Request served to the server-side scheduler code.

    Static fields are registered once at update_state_after_alloc. Fields that
    vLLM mutates in place on the live Request (num_computed_tokens, block_ids,
    all_token_ids) are refreshed per scheduling step from the
    SchedulerOutputView projection before the inherited business methods run.
    """

    request_id: str
    prompt_token_ids: list[int]
    block_hashes: list[BlockHash]
    num_prompt_tokens: int
    num_tokens: int
    num_computed_tokens: int = 0
    block_ids: tuple[list[int], ...] = ()
    # Prompt tokens plus generated tokens, synced incrementally per step.
    all_token_ids: list[int] = field(default_factory=list)

    @property
    def req_id(self) -> str:
        return self.request_id


@dataclass
class BlocksView:
    """Grouped block ids in place of a KVCacheBlocks allocation result."""

    block_ids_by_group: list[list[int]]

    def get_block_ids(self) -> tuple[list[int], ...]:
        return tuple(self.block_ids_by_group)


@dataclass
class RequestIdView:
    """Minimal request stand-in for callbacks that read only the id."""

    request_id: str


@dataclass
class ConnectorOutputView:
    """Stand-in for KVConnectorOutput: worker-side completion counts for the
    in-flight sending events, consumed by the inherited event accounting."""

    kv_connector_worker_meta: "AscendStoreKVConnectorWorkerMetadata"


@dataclass
class ScheduledNewReqPayload:
    """Serializable copy of the per-request fields vLLM carries in NewRequestData."""

    req_id: str
    num_computed_tokens: int
    block_ids_by_group: list[list[int]]


@dataclass
class CachedReqsView:
    """Server-side stand-in for ScheduledCachedReqs.

    Request ids and new blocks mirror the vLLM payload; num_computed_tokens
    and new_token_ids are the per-step dynamic fields refreshed onto the
    registered RequestViews.
    """

    req_ids: list[str]
    new_block_ids: list[list[list[int]] | None]
    num_computed_tokens: list[int]
    new_token_ids: dict[str, list[int]]


@dataclass
class SchedulerOutputView:
    """Server-side stand-in for SchedulerOutput.

    Elements of scheduled_new_reqs start as ScheduledNewReqPayload and are
    replaced with the refreshed RequestViews by _sync_request_views, so the
    inherited build_connector_meta reads a single object per new request.
    """

    finished_req_ids: set[str]
    preempted_req_ids: set[str]
    num_scheduled_tokens: dict[str, int]
    scheduled_new_reqs: list
    scheduled_cached_reqs: CachedReqsView
