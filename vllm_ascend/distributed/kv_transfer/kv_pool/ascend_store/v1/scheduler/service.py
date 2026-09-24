"""Scheduler-side Lookup and transfer decisions from the classic KV Pool path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import infer_group_block_sizes

from ..metadata import AscendStoreV1Metadata, LoadRequest, StoreRequest
from .load import LoadCandidate, LoadService
from .lookup import LookupService, SchedulerLookupRequest
from .store import StoreService

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class RequestTracker:
    req_id: str
    token_len: int
    block_ids: list[int]
    num_prompt_tokens: int

    def update(self, new_block_ids: tuple[list[int], ...] | None) -> None:
        if new_block_ids is not None:
            self.block_ids.extend(new_block_ids[0])


class SchedulerService:
    """Own request progress and assemble classic transfer metadata."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, lookup_address: str) -> None:
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)
        original_block_size = infer_group_block_sizes(
            vllm_config.cache_config.block_size, kv_cache_config.kv_cache_groups
        )[0]
        cache_transfer_granularity = original_block_size * dcp_size
        requested_hash_block_size = vllm_config.cache_config.prefix_match_unit
        hash_block_size = (
            requested_hash_block_size if isinstance(requested_hash_block_size, int) else original_block_size
        ) * dcp_size
        if cache_transfer_granularity % hash_block_size != 0:
            raise ValueError("block_size must be divisible by hash_block_size")
        discard_partial_chunks = extra_config.get("discard_partial_chunks", True)
        kv_role = vllm_config.kv_transfer_config.kv_role
        self._cache_transfer_granularity = cache_transfer_granularity
        self._hash_block_size = hash_block_size
        self._discard_partial_chunks = discard_partial_chunks
        self._lookup_service = LookupService(
            lookup_address,
            cache_transfer_granularity=cache_transfer_granularity,
            discard_partial_chunks=discard_partial_chunks,
            kv_role=kv_role,
            consumer_is_to_load=extra_config.get("consumer_is_to_load", False),
        )
        self._load_service = LoadService()
        self._store_service = StoreService(
            cache_transfer_granularity=cache_transfer_granularity,
            discard_partial_chunks=discard_partial_chunks,
            save_decode_cache=extra_config.get("save_decode_cache", False),
            kv_role=kv_role,
            consumer_is_to_put=extra_config.get("consumer_is_to_put", False),
        )
        self.request_trackers: dict[str, RequestTracker] = {}
        self.unfinished_requests: dict[str, Request] = {}
        self.preempted_req_ids: set[str] = set()

    def lookup(self, request: SchedulerLookupRequest) -> int:
        result = self._lookup_service.lookup(request)
        if result.kvpool_cached_tokens is not None:
            load_candidate = LoadCandidate(
                vllm_cached_tokens=request.num_computed_tokens,
                kvpool_cached_tokens=result.kvpool_cached_tokens,
            )
            self._load_service.record_candidate(request.req_id, load_candidate)
        return result.num_new_matched_tokens

    def update_state_after_alloc(self, request: Request, num_external_tokens: int) -> None:
        self.unfinished_requests[request.request_id] = request
        self._load_service.confirm_allocation(request.request_id, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> AscendStoreV1Metadata:
        for request_id in scheduler_output.finished_req_ids:
            self.request_trackers.pop(request_id, None)
            self.unfinished_requests.pop(request_id, None)
            self.preempted_req_ids.discard(request_id)
            self._store_service.discard(request_id)
        for request_id in scheduler_output.preempted_req_ids:
            self.preempted_req_ids.add(request_id)
            self.request_trackers.pop(request_id, None)
            self.unfinished_requests.pop(request_id, None)
            self._store_service.discard(request_id)

        metadata = AscendStoreV1Metadata(scheduler_output.preempted_req_ids)
        for scheduled_request in scheduler_output.scheduled_new_reqs:
            load_request, store_request = self._process_new_request(scheduled_request, scheduler_output)
            if load_request is not None:
                metadata.load_requests.append(load_request)
            if store_request is not None:
                metadata.store_requests.append(store_request)

        if self._store_service.can_store:
            cached_requests = scheduler_output.scheduled_cached_reqs
            for index, request_id in enumerate(cached_requests.req_ids):
                new_block_ids = cached_requests.new_block_ids[index]
                if not new_block_ids:
                    continue
                if request_id in self.preempted_req_ids:
                    load_request, store_request = self._process_preempted_cached_request(
                        request_id, new_block_ids, scheduler_output
                    )
                else:
                    load_request, store_request = self._process_running_cached_request(
                        request_id, new_block_ids, scheduler_output
                    )
                if load_request is not None:
                    metadata.load_requests.append(load_request)
                if store_request is not None:
                    metadata.store_requests.append(store_request)

        return metadata

    def _process_new_request(
        self, scheduled_request: NewRequestData, scheduler_output: SchedulerOutput
    ) -> tuple[LoadRequest | None, StoreRequest | None]:
        request_id = scheduled_request.req_id
        load_candidate = self._load_service.take_for_transfer(request_id)
        target_tokens = scheduled_request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request_id]
        request = self.unfinished_requests.get(request_id)
        if request is None:
            raise ValueError(
                f"Request {request_id} is not in _unfinished_requests, but it is scheduled as a new request"
            )
        tracker = RequestTracker(
            request_id, target_tokens, list(scheduled_request.block_ids[0]), len(request.prompt_token_ids)
        )
        self.request_trackers[request_id] = tracker
        return self._schedule_request_transfer(tracker, request, load_candidate)

    def _process_preempted_cached_request(
        self, request_id: str, new_block_ids: tuple[list[int], ...], scheduler_output: SchedulerOutput
    ) -> tuple[LoadRequest | None, StoreRequest | None]:
        self.preempted_req_ids.discard(request_id)
        load_candidate = self._load_service.take_for_transfer(request_id)
        request = self.unfinished_requests.get(request_id)
        if request is None:
            raise ValueError(
                f"Request {request_id} is not in _unfinished_requests, "
                "but it is scheduled as a preempted cached request"
            )
        target_tokens = request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request_id]
        tracker = RequestTracker(request_id, target_tokens, list(new_block_ids[0]), len(request.prompt_token_ids))
        self.request_trackers[request_id] = tracker
        return self._schedule_request_transfer(tracker, request, load_candidate)

    def _process_running_cached_request(
        self, request_id: str, new_block_ids: tuple[list[int], ...], scheduler_output: SchedulerOutput
    ) -> tuple[LoadRequest | None, StoreRequest | None]:
        request = self.unfinished_requests.get(request_id)
        is_decoding = request is not None and request.num_computed_tokens >= request.num_prompt_tokens
        if not self._store_service.accepts_cached_request(is_decoding):
            return None, None
        tracker = self.request_trackers.get(request_id)
        if tracker is None:
            raise ValueError(f"Request {request_id} is not in _request_trackers, but it is scheduled to be cached")
        if request is None:
            raise ValueError(f"Request {request_id} is not in _unfinished_requests, but it is scheduled to be cached")
        tracker.token_len += scheduler_output.num_scheduled_tokens[request_id]
        tracker.update(new_block_ids)
        return self._schedule_request_transfer(tracker, request, None)

    def _schedule_request_transfer(
        self, tracker: RequestTracker, request: Request, load_candidate: LoadCandidate | None
    ) -> tuple[LoadRequest | None, StoreRequest | None]:
        """Choose the operation and publish only its executable request."""
        transfer_end_token = self._resolve_transfer_end_token(tracker.token_len, len(request.block_hashes))
        if load_candidate is not None and load_candidate.allocation_confirmed:
            load_request = LoadRequest(
                request_id=tracker.req_id,
                transfer_end_token=transfer_end_token,
                block_ids=tuple(tracker.block_ids),
                block_hashes=tuple(request.block_hashes),
                vllm_cached_tokens=load_candidate.vllm_cached_tokens,
                kvpool_cached_tokens=load_candidate.kvpool_cached_tokens,
            )
            return load_request, None

        if not self._store_service.should_store(tracker.req_id, transfer_end_token):
            return None, None

        store_request = StoreRequest(
            request_id=tracker.req_id,
            save_end_token=transfer_end_token,
            block_ids=tuple(tracker.block_ids),
            block_hashes=tuple(request.block_hashes),
            num_prompt_tokens=tracker.num_prompt_tokens,
        )
        # This tracks scheduled Store work, not a confirmed Backend write.
        self._store_service.record_scheduled(tracker.req_id, store_request.save_end_token)
        return None, store_request

    def _resolve_transfer_end_token(self, target_token_len: int, num_block_hashes: int) -> int:
        transfer_end_token = (
            target_token_len // self._cache_transfer_granularity * self._cache_transfer_granularity
            if self._discard_partial_chunks
            else target_token_len
        )
        hashes_per_transfer_block = self._cache_transfer_granularity // self._hash_block_size
        full_block_count = target_token_len // self._cache_transfer_granularity
        available_full_block_count = num_block_hashes // hashes_per_transfer_block
        boundary_without_hash = (
            target_token_len > 0
            and target_token_len % self._cache_transfer_granularity == 0
            and full_block_count > available_full_block_count
        )
        if boundary_without_hash:
            return available_full_block_count * self._cache_transfer_granularity
        return transfer_end_token

    def close(self) -> None:
        self._lookup_service.close()
