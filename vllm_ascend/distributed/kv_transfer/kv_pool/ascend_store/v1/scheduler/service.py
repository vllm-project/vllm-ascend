"""Scheduler-side Lookup and transfer decisions from the classic KV Pool path."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..metadata import AscendStoreV1Metadata, LoadRequest, LoadRequestBatch, StoreRequest, StoreRequestBatch
from .layout import SchedulerTransferLayout
from .load import LoadCandidate, LoadService
from .lookup import LookupService, SchedulerLookupRequest
from .request_tracker import RequestTracker
from .store import StoreService

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.request import Request


class SchedulerService:
    """Own request progress and assemble classic transfer metadata."""

    def __init__(
        self,
        layout: SchedulerTransferLayout,
        lookup_service: LookupService,
        load_service: LoadService,
        store_service: StoreService,
    ) -> None:
        self._layout = layout
        self._lookup_service = lookup_service
        self._load_service = load_service
        self._store_service = store_service
        self.request_trackers: dict[str, RequestTracker] = {}
        self.unfinished_requests: dict[str, Request] = {}
        self.preempted_req_ids: set[str] = set()

    def lookup(self, request: SchedulerLookupRequest) -> tuple[int, bool]:
        result = self._lookup_service.lookup(request)
        if result.kvpool_cached_tokens is not None:
            load_candidate = LoadCandidate(
                vllm_cached_tokens=request.num_computed_tokens,
                kvpool_cached_tokens=result.kvpool_cached_tokens,
            )
            self._load_service.record_candidate(request.req_id, load_candidate)
        load_is_deferred = result.num_new_matched_tokens > 0 and self._load_service.is_deferred
        return result.num_new_matched_tokens, load_is_deferred

    def update_state_after_alloc(
        self, request: Request, blocks: tuple[list[int], ...], num_external_tokens: int
    ) -> None:
        request_id = request.request_id
        self.unfinished_requests[request_id] = request
        candidate = self._load_service.confirm_allocation(request_id, num_external_tokens)
        if candidate is None:
            return

        target_tokens = candidate.kvpool_cached_tokens
        granularity = self._layout.cache_transfer_granularity
        if target_tokens % granularity != 0 and target_tokens == len(request.prompt_token_ids) - 1:
            target_tokens += 1
        block_ids = list(blocks[0])
        num_prompt_tokens = len(request.prompt_token_ids)
        self.request_trackers[request_id] = RequestTracker(
            request_id, target_tokens, block_ids, request.block_hashes, num_prompt_tokens
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> AscendStoreV1Metadata:
        self._handle_finished_and_preempted_requests(scheduler_output)
        load_requests, store_requests = self._build_transfers_for_scheduled_requests(scheduler_output)
        load_requests.extend(self._build_load_requests_for_ready_candidates())

        return AscendStoreV1Metadata(
            LoadRequestBatch(tuple(load_requests)),
            StoreRequestBatch(tuple(store_requests), frozenset(scheduler_output.preempted_req_ids)),
        )

    def _handle_finished_and_preempted_requests(self, scheduler_output: SchedulerOutput) -> None:
        for request_id in scheduler_output.finished_req_ids:
            self.request_trackers.pop(request_id, None)
            self.unfinished_requests.pop(request_id, None)
            self.preempted_req_ids.discard(request_id)
            self._load_service.discard_transfer(request_id)
            self._store_service.discard(request_id)
        for request_id in scheduler_output.preempted_req_ids:
            self.preempted_req_ids.add(request_id)
            self.request_trackers.pop(request_id, None)
            self.unfinished_requests.pop(request_id, None)
            self._load_service.discard_transfer(request_id)
            self._store_service.discard(request_id)

    def _build_transfers_for_scheduled_requests(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[LoadRequest], list[StoreRequest]]:
        load_requests: list[LoadRequest] = []
        store_requests: list[StoreRequest] = []
        for scheduled_request in scheduler_output.scheduled_new_reqs:
            load_request, store_request = self._process_new_request(scheduled_request, scheduler_output)
            if load_request is not None:
                load_requests.append(load_request)
            if store_request is not None:
                store_requests.append(store_request)

        if self._store_service.is_enabled:
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
                    load_requests.append(load_request)
                if store_request is not None:
                    store_requests.append(store_request)

        return load_requests, store_requests

    def _build_load_requests_for_ready_candidates(self) -> list[LoadRequest]:
        load_requests: list[LoadRequest] = []
        for request_id, load_candidate in self._load_service.take_ready_for_transfer():
            request = self.unfinished_requests.get(request_id)
            tracker = self.request_trackers.get(request_id)
            if request is None or tracker is None:
                raise ValueError(f"Request {request_id} is ready for asynchronous Load without allocated blocks")
            load_request, store_request = self._schedule_request_transfer(tracker, load_candidate)
            if load_request is None or store_request is not None:
                raise ValueError(f"Request {request_id} did not produce an asynchronous Load request")
            load_requests.append(load_request)

        return load_requests

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
            request_id,
            target_tokens,
            list(scheduled_request.block_ids[0]),
            request.block_hashes,
            len(request.prompt_token_ids),
        )
        self.request_trackers[request_id] = tracker
        return self._schedule_request_transfer(tracker, load_candidate)

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
        tracker = RequestTracker(
            request_id,
            target_tokens,
            list(new_block_ids[0]),
            request.block_hashes,
            len(request.prompt_token_ids),
        )
        self.request_trackers[request_id] = tracker
        return self._schedule_request_transfer(tracker, load_candidate)

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
        tracker.advance(scheduler_output.num_scheduled_tokens[request_id], new_block_ids, request.block_hashes)
        return self._schedule_request_transfer(tracker, None)

    def _schedule_request_transfer(
        self, tracker: RequestTracker, load_candidate: LoadCandidate | None
    ) -> tuple[LoadRequest | None, StoreRequest | None]:
        """Choose the operation and publish only its executable request."""
        transfer_end_token = self._resolve_transfer_end_token(tracker.token_len, len(tracker.block_hashes))
        if load_candidate is not None:
            load_request = self._load_service.schedule_request(tracker, transfer_end_token, load_candidate)
            return load_request, None

        store_request = self._store_service.schedule_request(tracker, transfer_end_token)
        return None, store_request

    def _resolve_transfer_end_token(self, target_token_len: int, num_block_hashes: int) -> int:
        granularity = self._layout.cache_transfer_granularity
        transfer_end_token = target_token_len
        if self._layout.discard_partial_chunks:
            transfer_end_token = target_token_len // granularity * granularity
        hashes_per_transfer_block = granularity // self._layout.hash_block_size
        full_block_count = target_token_len // granularity
        available_full_block_count = num_block_hashes // hashes_per_transfer_block
        boundary_without_hash = (
            target_token_len > 0
            and target_token_len % granularity == 0
            and full_block_count > available_full_block_count
        )
        if boundary_without_hash:
            return available_full_block_count * granularity
        return transfer_end_token

    def close(self) -> None:
        self._lookup_service.close()
