"""Scheduler request lifecycle and Bulk/Layerwise Load publication policies."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from vllm.utils.math_utils import cdiv

from ...metrics import AscendStoreKVConnectorStats
from ..infrastructure.configuration import KVPoolConfig
from ..protocol.lookup import LookupClient
from ..protocol.messages import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    LookupRequest,
    LookupResult,
    RangeStoreCommand,
    StoreCommandBatch,
    StoreSourceReleaseMetadata,
)
from .state import LoadCandidate, RequestAllocation, RequestProgress, RequestState
from .store import StorePlanner, StoreSourceLeases

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class KVPoolScheduler:
    """Own Scheduler-side request progress and publish Worker transfer steps."""

    load_is_deferred = False
    store_with_scheduled_load = False

    def __init__(self, config: KVPoolConfig, lookup_client: LookupClient) -> None:
        self._config = config
        self._lookup_client = lookup_client
        self._request_states: dict[str, RequestState] = {}
        # An ordered publication index; candidates and allocation facts live only in RequestState.
        self._ready_loads: dict[str, RequestState] = {}
        self._finished_checkpoint_stores: list[CheckpointStoreCommand] = []
        self._store_planner = StorePlanner(config)
        self._source_leases = StoreSourceLeases(config)
        self._has_state_checkpoints = config.store_enabled and not config.align_state_group_ids.isdisjoint(
            config.transfer_group_ids
        )

    # ================================
    # Lookup and Allocation
    # ================================

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        state = self._request_states.get(request.request_id)
        if state is not None:
            state.pending_lookup = None
        query_end = request.num_prompt_tokens
        query_end -= query_end % self._config.cache_transfer_granularity
        local_available = min(num_computed_tokens, query_end)
        if (
            not self._config.load_enabled
            or query_end < self._config.cache_transfer_granularity
            or local_available >= query_end
        ):
            if state is not None and state.allocation is None:
                self._discard_request(request.request_id)
            return 0, False
        if state is None:
            state = self._request_states[request.request_id] = RequestState(request)

        result = self._lookup_client.lookup(
            LookupRequest(
                local_available, query_end, self._config.transfer_group_ids, state.snapshot_hashes(request.block_hashes)
            )
        )
        candidate = self._resolve_load_candidate(
            result,
            query_end_token=query_end,
            local_cached_tokens=num_computed_tokens,
            request_token_count=request.num_tokens,
        )
        if candidate is None:
            return 0, False
        state.pending_lookup = candidate
        num_new_matched_tokens = candidate.num_new_matched_tokens
        return num_new_matched_tokens, self.load_is_deferred and num_new_matched_tokens > 0

    def confirm_allocation(self, request: Request, blocks: KVCacheBlocks, allocated_external_tokens: int) -> None:
        request_id = request.request_id
        state = self._request_states.get(request_id)
        candidate = None if state is None else state.pending_lookup
        selected_for_load = False
        if candidate is not None:
            expected_external_tokens = candidate.num_new_matched_tokens
            if allocated_external_tokens not in (0, expected_external_tokens):
                raise ValueError(
                    f"Allocation confirmed {allocated_external_tokens} external tokens for request {request_id}; "
                    f"expected either 0 or {expected_external_tokens}"
                )
            selected_for_load = expected_external_tokens > 0 and allocated_external_tokens == expected_external_tokens

        if not self._config.store_enabled and not selected_for_load and (state is None or state.pending_load is None):
            self._discard_request(request_id)
            return
        block_ids_by_group = _freeze_block_ids(blocks.get_block_ids())
        if state is None:
            state = self._request_states[request_id] = RequestState(request)
        block_hashes = state.snapshot_hashes(request.block_hashes)
        previous = state.allocation
        if candidate is not None:
            store_skip_end = candidate.store_skip_end_token
        else:
            store_skip_end = 0 if previous is None else previous.store_skip_end_token
        state.request = request
        state.allocation = RequestAllocation(block_ids_by_group, block_hashes, store_skip_end)
        state.pending_lookup = None
        if selected_for_load:
            state.pending_load = candidate
            self._ready_loads[request_id] = state

    def _resolve_load_candidate(
        self,
        lookup_result: LookupResult,
        *,
        query_end_token: int,
        local_cached_tokens: int,
        request_token_count: int,
    ) -> LoadCandidate | None:
        """Derive readable, Store-skip, and Scheduler-safe prefix frontiers."""
        config = self._config
        readable_end = lookup_result.available_end_token
        if readable_end == 0 or readable_end <= local_cached_tokens:
            return None
        if readable_end > query_end_token:
            raise ValueError(f"Lookup returned token {readable_end} beyond query end {query_end_token}")

        store_skip_end = readable_end
        safe_load_end = readable_end
        granularity = config.cache_transfer_granularity
        if config.use_eagle and request_token_count > 0:
            final_block_start = (request_token_count - 1) // granularity * granularity
            if safe_load_end > final_block_start:
                safe_load_end = max(local_cached_tokens, safe_load_end - granularity)
        if safe_load_end == request_token_count:
            safe_load_end -= 1
        if config.has_private_state:
            safe_load_end = safe_load_end // granularity * granularity
        safe_load_end = max(local_cached_tokens, safe_load_end)

        physical_load_end = min(readable_end, cdiv(safe_load_end, granularity) * granularity)
        load_start = local_cached_tokens
        if physical_load_end < load_start:
            physical_load_end = load_start
        # Lookup may select a longer Attention object in the same physical tail page.
        # Its identity remains valid unless the physical Load frontier changes.
        tail_boundaries = lookup_result.tail_key_boundaries
        if physical_load_end != readable_end:
            tail_boundaries = tuple(
                boundary for boundary in tail_boundaries if boundary.boundary_token <= physical_load_end
            )
        return LoadCandidate(
            local_cached_tokens,
            store_skip_end,
            safe_load_end,
            load_start,
            physical_load_end,
            tail_boundaries,
        )

    # ================================
    # Scheduler Step Publication
    # ================================

    def build_step(self, scheduler_output: SchedulerOutput) -> KVTransferStep:
        # Termination is already a vLLM fact, even if preparing this step later fails.
        finished_request_ids = frozenset(scheduler_output.finished_req_ids)
        preempted_request_ids = frozenset(scheduler_output.preempted_req_ids or ())
        for request_id in finished_request_ids | preempted_request_ids:
            self._discard_request(request_id)

        load_commands: list[LoadCommand] = []
        source_pending_commands: list[RangeStoreCommand] = []
        source_ready_commands: list[CheckpointStoreCommand] = []
        progress_updates: dict[str, RequestProgress] = {}
        load_request_ids: set[str] = set()

        # Prepare final commands and tentative progress without consuming any pending Load.
        for scheduled in scheduler_output.scheduled_new_reqs:
            if not self._config.store_enabled and scheduled.req_id not in self._ready_loads:
                continue
            state = self._allocated_request(scheduled.req_id)
            progress, load_command, store_command = self._plan_admitted_request(
                state,
                scheduled.block_ids,
                scheduled.num_computed_tokens,
                scheduler_output.num_scheduled_tokens[scheduled.req_id],
                len(scheduled.prefill_token_ids)
                if getattr(scheduled, "prefill_token_ids", None) is not None
                else state.request.num_prompt_tokens,
            )
            if progress is not None:
                progress_updates[scheduled.req_id] = progress
            if state.pending_load is not None:
                load_request_ids.add(scheduled.req_id)
            if load_command is not None:
                load_commands.append(load_command)
            if store_command is not None:
                source_pending_commands.append(store_command)

        cached = scheduler_output.scheduled_cached_reqs
        resumed_request_ids = frozenset(cached.resumed_req_ids)
        for index, request_id in enumerate(cached.req_ids):
            if not self._config.store_enabled and request_id not in self._ready_loads:
                continue
            block_ids = cached.new_block_ids[index]
            if request_id in resumed_request_ids:
                if block_ids is None:
                    raise ValueError(f"Scheduled RESUMED request {request_id} has no replacement block table")
                state = self._allocated_request(request_id)
                progress, load_command, store_command = self._plan_admitted_request(
                    state,
                    block_ids,
                    cached.num_computed_tokens[index],
                    scheduler_output.num_scheduled_tokens[request_id],
                    state.request.num_tokens,
                )
                if state.pending_load is not None:
                    load_request_ids.add(request_id)
            else:
                running_progress, store_command = self._plan_running_request(
                    request_id,
                    block_ids,
                    cached.num_computed_tokens[index],
                    scheduler_output.num_scheduled_tokens[request_id],
                )
                if running_progress is None:
                    continue
                progress = running_progress
                load_command = None
            if progress is not None:
                progress_updates[request_id] = progress
            if load_command is not None:
                load_commands.append(load_command)
            if store_command is not None:
                source_pending_commands.append(store_command)

        # Deferred Loads can publish without a scheduled forward, including on an empty step.
        if self.load_is_deferred:
            for request_id, state in self._ready_loads.items():
                if request_id in load_request_ids:
                    continue
                command = self._make_load_command(state)
                load_request_ids.add(request_id)
                if command is not None:
                    load_commands.append(command)

        block_state = scheduler_output.kv_connector_block_state
        if self._config.store_enabled and block_state is not None:
            for request_id, entries in block_state.boundary_state_offloads.items():
                checkpoint_state = self._request_states.get(request_id)
                if checkpoint_state is None:
                    continue
                checkpoint_progress = progress_updates.get(request_id, checkpoint_state.progress)
                if checkpoint_progress is not None:
                    source_ready_commands.extend(
                        self._store_planner.plan_checkpoints(checkpoint_state.request, checkpoint_progress, entries)
                    )
        source_ready_commands.extend(self._finished_checkpoint_stores)

        # Acquire source ownership last; failure leaves plans and existing checkpoint leases pending.
        store = self._source_leases.acquire_step(
            StoreCommandBatch(tuple(source_pending_commands), tuple(source_ready_commands))
        )
        step = KVTransferStep(
            tuple(load_commands),
            store,
            finished_request_ids | preempted_request_ids,
        )
        for request_id, progress in progress_updates.items():
            self._request_states[request_id].progress = progress
        for request_id in load_request_ids:
            self._request_states[request_id].pending_load = None
            self._ready_loads.pop(request_id, None)
        self._finished_checkpoint_stores.clear()
        return step

    def _plan_admitted_request(
        self,
        state: RequestState,
        block_ids_by_group: Sequence[Sequence[int]],
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        prefill_end_token: int,
    ) -> tuple[RequestProgress | None, LoadCommand | None, RangeStoreCommand | None]:
        request = state.request
        allocation = state.allocation
        assert allocation is not None
        request_id = request.request_id
        if len(block_ids_by_group) != len(allocation.block_ids_by_group) or any(
            tuple(actual) != confirmed
            for actual, confirmed in zip(block_ids_by_group, allocation.block_ids_by_group, strict=True)
        ):
            raise ValueError(f"Scheduled request {request_id} does not match its confirmed allocation")
        load_command = self._make_load_command(state)
        if not self._config.store_enabled:
            return None, load_command, None
        scheduled_end = num_computed_tokens + num_scheduled_tokens
        progress = RequestProgress(
            request_id,
            scheduled_end,
            allocation.block_ids_by_group,
            state.snapshot_hashes(request.block_hashes),
            prefill_end_token,
            allocation.store_skip_end_token,
            min(scheduled_end, request.num_tokens),
        )
        store_command = None
        if load_command is None or self.store_with_scheduled_load:
            progress, store_command = self._store_planner.plan_range(request, progress)
        return progress, load_command, store_command

    def _plan_running_request(
        self,
        request_id: str,
        new_block_ids: Sequence[Sequence[int]] | None,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> tuple[RequestProgress | None, RangeStoreCommand | None]:
        if not self._config.store_enabled:
            return None, None
        state = self._request_states.get(request_id)
        if state is None or state.progress is None:
            raise ValueError(f"Scheduled RUNNING request {request_id} has no Scheduler progress")
        request, progress = state.request, state.progress

        is_decoding = num_computed_tokens >= progress.prefill_end_token
        if is_decoding and not self._config.save_decode_cache and not self._has_state_checkpoints:
            # No decode transfer consumes these snapshots. Resumption rebuilds
            # them from the confirmed allocation; state checkpoints still need progress.
            return None, None

        scheduled_end = num_computed_tokens + num_scheduled_tokens
        progress = progress.advance(
            scheduled_end,
            new_block_ids,
            state.snapshot_hashes(request.block_hashes),
            min(scheduled_end, request.num_tokens),
        )
        store_command = None
        if not is_decoding or self._config.save_decode_cache:
            progress, store_command = self._store_planner.plan_range(request, progress)
        return progress, store_command

    @staticmethod
    def _make_load_command(state: RequestState) -> LoadCommand | None:
        candidate = state.pending_load
        if candidate is None:
            return None
        request_id = state.request.request_id
        allocation = state.allocation
        if allocation is None:
            raise ValueError(f"Request {request_id} is ready for Load without a confirmed allocation")
        if candidate.load_end_token <= candidate.load_start_token:
            return None
        return LoadCommand(
            request_id,
            candidate.load_start_token,
            candidate.load_end_token,
            allocation.block_ids_by_group,
            allocation.block_hashes,
            candidate.tail_key_boundaries,
        )

    # ================================
    # Store Sources and Results
    # ================================

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        state = self._request_states.get(request.request_id)
        if state is None or state.progress is None:
            return False
        command = self._store_planner.plan_finished_checkpoint(
            request, state.progress, block_ids, partial_tail_offloads
        )
        if command is not None:
            self._finished_checkpoint_stores.append(self._source_leases.acquire(command))
        # vLLM may release its request object; the exact source blocks remain leased.
        return False

    def bind_gpu_block_pool(self, block_pool: BlockPool) -> None:
        self._source_leases.bind_block_pool(block_pool)

    def accept_worker_metadata(self, metadata: Any) -> None:
        if isinstance(metadata, StoreSourceReleaseMetadata):
            self._source_leases.release(metadata.released_store_jobs)

    def has_pending_push_work(self) -> bool:
        return self._source_leases.has_pending() or bool(self._finished_checkpoint_stores)

    def has_pending_block_frees(self) -> bool:
        return self._source_leases.has_pending_block_frees()

    def get_stats(self) -> AscendStoreKVConnectorStats:
        stats = AscendStoreKVConnectorStats()
        jobs, blocks = self._source_leases.pending_counts()
        stats.set_delayed_release(jobs, blocks)
        return stats

    def reset_cache(self) -> bool | None:
        """Reset only a quiescent supported route; caller must pause generation first."""
        if self._config.backend_name != "mooncake":
            return None
        if self._config.use_layerwise or self.has_pending_push_work() or self._ready_loads:
            return False
        # A failed reply cannot prove that the remote reset did not execute.
        for request_id, state in tuple(self._request_states.items()):
            state.pending_lookup = None
            if state.allocation is None:
                del self._request_states[request_id]
        return self._lookup_client.reset()

    # ================================
    # Request and Scheduler Lifecycle
    # ================================

    def _allocated_request(self, request_id: str) -> RequestState:
        state = self._request_states.get(request_id)
        if state is None or state.allocation is None:
            raise ValueError(f"Scheduled request {request_id} has not passed allocation confirmation")
        return state

    def _discard_request(self, request_id: str) -> None:
        self._request_states.pop(request_id, None)
        self._ready_loads.pop(request_id, None)

    def close(self) -> None:
        self._lookup_client.close()
        pending_job_ids = self._source_leases.pending_job_ids()
        queued_checkpoints = len(self._finished_checkpoint_stores)
        if pending_job_ids or queued_checkpoints:
            raise RuntimeError(
                "Cannot close KV Pool Scheduler with pending Store ownership: "
                f"source lease job ids={pending_job_ids}, queued finished checkpoints={queued_checkpoints}"
            )


def _freeze_block_ids(block_ids_by_group: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(block_ids) for block_ids in block_ids_by_group)


class SynchronousScheduler(KVPoolScheduler):
    """Publish a Load when its allocated request is scheduled."""

    store_with_scheduled_load = True


class AsynchronousScheduler(KVPoolScheduler):
    """Publish allocated Bulk Loads even before their requests are scheduled."""

    load_is_deferred = True
