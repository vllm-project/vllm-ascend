from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata, KVConnectorWorkerMetadata
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    TPMismatchInfo as TPMismatchInfo,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    get_group_block_size as get_group_block_size,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    get_group_cache_family as get_group_cache_family,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    infer_cache_family_from_ratio as infer_cache_family_from_ratio,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    infer_cache_transfer_granularity as infer_cache_transfer_granularity,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    infer_cacheable_group_ids as infer_cacheable_group_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    infer_dcp_mismatch_info as infer_dcp_mismatch_info,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    infer_group_block_sizes as infer_group_block_sizes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    infer_group_cache_families as infer_group_cache_families,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    infer_tp_mismatch_info as infer_tp_mismatch_info,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    is_kv_save_role as is_kv_save_role,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_config import (
    uses_hybrid_kv_cache as uses_hybrid_kv_cache,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    KeyMetadata as KeyMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    PoolKey as PoolKey,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    block_hash_to_bytes as block_hash_to_bytes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    block_hash_to_str as block_hash_to_str,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    get_block_hashes as get_block_hashes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.request_state import (
    RequestTracker as RequestTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.request_state import (
    normalize_block_ids_by_group as normalize_block_ids_by_group,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.token_database import (
    ChunkedTokenDatabase as ChunkedTokenDatabase,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.transfer_metadata import RequestTransferState


# Parameters related to the connector metadata
@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in kvpool
    kvpool_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool
    # Raw KVPool hit length used to avoid storing an already pooled prefix.
    kvpool_store_skip_tokens: int | None = None

    token_len: int = 0

    # Exact request-scoped snapshot identity, separate from the logical length
    # which can roll back after speculative-token rejection.
    layerwise_snapshot_end_token: int | None = None
    layerwise_snapshot_block_index: int | None = None
    # Distinguish the preceding step of this request from external prefix hits.
    layerwise_previous_step: bool = False
    # First block stored under request-scoped keys, including unhashed full blocks.
    layerwise_snapshot_start_block: int | None = None


@dataclass(init=False)
class ReqMeta:
    # Request id
    req_id: str
    token_len_chunk: int
    # End token for full-block KV save.
    save_end_token: int
    # Token length after this scheduled step finishes.
    target_token_len: int
    block_ids_by_group: list[list[int]]

    block_hashes: list[BlockHash]

    # First token that has not been saved before this metadata was built.
    save_start_token: int = 0

    can_save: bool | None = None
    # load_spec
    load_spec: LoadSpec | None = None

    is_last_chunk: bool | None = None

    current_event: torch.npu.Event | None = None
    kv_cache_group_ids: list[int] | None = None
    skip_null_blocks_by_group: list[bool] | None = None
    num_prompt_tokens: int | None = None

    # The following parameters are only used for kv event generation
    # TODO: add lora_request which used for gen lora_id/lora_name in kv event
    token_ids: list[int] | None = None
    original_block_size: list[int] | int | None = None

    event_id: int | None = None
    # Monotonic identity of one queued store operation. Unlike req_id, this
    # value is never reused while the engine is alive, so an old asynchronous
    # operation cannot update the save progress of a new request generation
    # that happens to reuse the same req_id.
    store_job_id: int | None = None
    transfer: RequestTransferState = field(default_factory=RequestTransferState)

    def __init__(
        self,
        req_id: str,
        token_len_chunk: int | None = None,
        block_ids_by_group: list[list[int]] | None = None,
        block_hashes: list[BlockHash] | None = None,
        can_save: bool | None = None,
        load_spec: LoadSpec | None = None,
        is_last_chunk: bool | None = None,
        current_event: torch.npu.Event | None = None,
        kv_cache_group_ids: list[int] | None = None,
        skip_null_blocks_by_group: list[bool] | None = None,
        num_prompt_tokens: int | None = None,
        token_ids: list[int] | None = None,
        original_block_size: list[int] | int | None = None,
        block_ids: list[int] | list[list[int]] | None = None,
        event_id: int | None = None,
        store_job_id: int | None = None,
        save_end_token: int | None = None,
        target_token_len: int | None = None,
        save_start_token: int = 0,
        last_block_gva: int | None = None,
        partial_block_index: int | None = None,
        transfer: RequestTransferState | None = None,
        **transfer_fields: Any,
    ) -> None:
        if transfer is not None and transfer_fields:
            raise TypeError("Use transfer or legacy transfer fields, not both")
        self.transfer = transfer if transfer is not None else RequestTransferState.from_legacy_fields(**transfer_fields)
        if token_len_chunk is None:
            token_len_chunk = 0 if save_end_token is None else save_end_token
        self.req_id = req_id
        self.token_len_chunk = token_len_chunk
        self.save_end_token = token_len_chunk if save_end_token is None else save_end_token
        self.target_token_len = token_len_chunk if target_token_len is None else target_token_len
        self.save_start_token = save_start_token
        if block_ids_by_group is None:
            block_ids_by_group = normalize_block_ids_by_group(block_ids or [])
        self.block_ids_by_group = block_ids_by_group
        self.block_hashes = [] if block_hashes is None else block_hashes
        self.can_save = can_save
        self.load_spec = load_spec
        self.is_last_chunk = is_last_chunk
        self.current_event = current_event
        self.kv_cache_group_ids = kv_cache_group_ids
        self.skip_null_blocks_by_group = skip_null_blocks_by_group
        self.num_prompt_tokens = num_prompt_tokens
        self.token_ids = token_ids
        self.original_block_size = original_block_size
        self.event_id = event_id
        self.store_job_id = store_job_id
        self.last_block_gva = last_block_gva
        self.partial_block_index = partial_block_index

    def __copy__(self) -> ReqMeta:
        # Group views rebind prepared fields independently, while retaining the
        # same shallow ownership of underlying arrays/lists as the old payload.
        view = object.__new__(type(self))
        view.__dict__ = self.__dict__.copy()
        view.transfer = copy(self.transfer)
        return view

    @property
    def block_ids(self) -> list[int]:
        return self.block_ids_by_group[0] if self.block_ids_by_group else []

    @block_ids.setter
    def block_ids(self, block_ids: list[int] | list[list[int]]) -> None:
        self.block_ids_by_group = normalize_block_ids_by_group(block_ids)

    def get_event_token_ids(self, start: int, end: int) -> list[int] | None:
        """Return token ids for an absolute request range.

        Metadata carries only the newly savable suffix to avoid repeatedly
        serializing the full request history during long decode sequences.
        An empty list means this metadata does not cover the requested range.
        """
        if self.token_ids is None:
            return None
        token_ids_end = self.save_start_token + len(self.token_ids)
        if start < self.save_start_token or end > token_ids_end:
            return []
        return self.token_ids[start - self.save_start_token : end - self.save_start_token]

    def get_block_ids_array(self, group_id: int) -> np.ndarray | None:
        return self.transfer.get_block_ids_array(group_id)

    def get_transfer_arrays(self, group_id: int, is_save: bool = True) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.transfer.get_transfer_arrays(group_id, is_save)

    last_block_gva: int | None = None
    partial_block_index: int | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        cache_transfer_granularity: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
        discard_partial_chunks: bool = True,
        original_block_size: list[int] | int | None = None,
        save_partial_block: bool = False,
        hash_block_size: int | None = None,
    ) -> ReqMeta | None:
        """Apply this step's save decision and snapshot the connector payload."""
        block_hashes = [] if block_hashes is None else block_hashes
        decision = tracker.plan_save(
            cache_transfer_granularity,
            len(block_hashes),
            can_load=load_spec is not None and load_spec.can_load,
            skip_save=skip_save,
            discard_partial_chunks=discard_partial_chunks,
            save_partial_block=save_partial_block,
            hash_block_size=hash_block_size,
        )
        if decision.skip_save and load_spec is None:
            return None
        tracker.apply_save_decision(decision)
        token_ids = None
        if tracker.token_ids and not decision.skip_save:
            # The async worker owns only the newly savable suffix.
            token_ids_end = decision.target_token_len if decision.should_save_partial_block else decision.save_end_token
            token_ids = tracker.token_ids[decision.previous_saved_tokens : token_ids_end]
        if load_spec is not None and load_spec.can_load:
            logger.debug("Scheduled to load %d tokens for request %s", load_spec.kvpool_cached_tokens, tracker.req_id)
        else:
            load_spec = None
        logger.debug(
            "request:%s, meta save spec:%s, meta load spec:%s", tracker.req_id, not decision.skip_save, load_spec
        )
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=decision.save_end_token,
            save_end_token=decision.save_end_token,
            target_token_len=decision.target_token_len,
            save_start_token=decision.previous_saved_tokens,
            block_ids_by_group=tracker.allocated_block_ids_by_group,
            can_save=not decision.skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
            token_ids=token_ids,
            # A resumed MRV2 request re-prefills prompt and generated tokens.
            num_prompt_tokens=tracker.prefill_end_tokens or tracker.num_prompt_tokens or decision.target_token_len,
            original_block_size=original_block_size,
            last_block_gva=tracker.last_block_gva,
            partial_block_index=decision.partial_block_index,
            transfer=RequestTransferState.from_tracker(tracker),
            kv_cache_group_ids=list(range(len(tracker.allocated_block_ids_by_group))),
        )


class AscendConnectorMetadata(KVConnectorMetadata):
    def __init__(
        self,
        preempted_req_ids,
        loading_req_ids: set[str] | None = None,
    ):
        self.requests: list[ReqMeta] = []
        self.preempted_req_ids = preempted_req_ids
        self.loading_req_ids = loading_req_ids or set()
        # Requests whose blocks became reusable since the previous step
        # (finished + preempted); the worker fences their in-flight saves.
        self.released_req_ids: set[str] = set(preempted_req_ids or ())

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata."""
        self.requests.append(req_meta)


@dataclass
class AscendStoreKVConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    completed_events: dict[int, int] = field(default_factory=dict)
    """key: event_id, value: completed worker count"""

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, AscendStoreKVConnectorWorkerMetadata), (
            "aggregate worker metadata must be type of AscendStoreKVConnectorWorkerMetadata"
        )

        merged: dict[int, int] = dict(self.completed_events)
        for event_id in other.completed_events:
            if event_id not in merged:
                merged[event_id] = other.completed_events[event_id]
            else:
                merged[event_id] = merged[event_id] + other.completed_events[event_id]
        return AscendStoreKVConnectorWorkerMetadata(merged)
