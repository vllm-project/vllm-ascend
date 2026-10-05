"""Shared Scheduler mainline for Lookup, publication, and source ownership."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypeVar

from vllm.utils.math_utils import cdiv

from ..coordinates import TokenRange
from ..protocol.transfer import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    LoadCommandBatch,
    RangeStoreCommand,
    StateCheckpointSource,
    StoreCommandBatch,
    StoreSourceReleaseMetadata,
)
from .lookup import RemoteLookup, resolve_load_candidate
from .source_leases import StoreSourceLeases
from .state import LoadCandidate, RequestAllocation, RequestProgress, SchedulerConfig

CommandT = TypeVar("CommandT")

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class KVPoolScheduler:
    """Own Scheduler-side request progress and publish Worker transfer steps."""

    load_is_deferred = False
    store_with_scheduled_load = False

    def __init__(self, config: SchedulerConfig, remote_lookup: RemoteLookup) -> None:
        self._config = config
        self._remote_lookup = remote_lookup
        self._pending_lookups: dict[str, LoadCandidate] = {}
        self._requests: dict[str, Request] = {}
        self._allocations: dict[str, RequestAllocation] = {}
        self._request_progress: dict[str, RequestProgress] = {}
        self._finished_checkpoint_stores: list[CheckpointStoreCommand] = []
        self._source_leases = StoreSourceLeases(
            frozenset(config.transfer_group_ids),
            config.align_state_group_ids,
            config.expected_worker_count,
        )

    # ================================
    # Lookup and Allocation
    # ================================

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        if not self._config.load_enabled:
            return 0, False

        query_end = request.num_prompt_tokens
        if self._config.discard_partial_chunks:
            query_end -= query_end % self._config.cache_transfer_granularity
        minimum_query_tokens = (
            self._config.cache_transfer_granularity
            if self._config.discard_partial_chunks
            else self._config.hash_block_size
        )
        if query_end < minimum_query_tokens:
            return 0, False
        query_start = self._lookup_start_token(num_computed_tokens, query_end)
        if query_start >= query_end:
            return 0, False

        query_range = TokenRange(query_start, query_end)
        result = self._remote_lookup.query(query_range, self._config.transfer_group_ids, tuple(request.block_hashes))
        candidate = resolve_load_candidate(
            self._config,
            result,
            query_range=query_range,
            local_cached_tokens=num_computed_tokens,
            request_token_count=request.num_tokens,
        )
        if candidate is None:
            self._pending_lookups.pop(request.request_id, None)
            return 0, False
        self._pending_lookups[request.request_id] = candidate
        num_new_matched_tokens = candidate.num_new_matched_tokens
        return num_new_matched_tokens, self.load_is_deferred and num_new_matched_tokens > 0

    def confirm_allocation(self, request: Request, blocks: KVCacheBlocks, allocated_external_tokens: int) -> None:
        request_id = request.request_id
        block_ids_by_group = _freeze_block_ids(blocks.get_block_ids())
        block_hashes = tuple(request.block_hashes)
        candidate = self._pending_lookups.pop(request_id, None)
        previous = self._allocations.get(request_id)
        if candidate is None and previous is not None:
            self._requests[request_id] = request
            self._allocations[request_id] = RequestAllocation(
                block_ids_by_group,
                block_hashes,
                previous.store_skip_end_token,
            )
            return

        store_skip_end = 0 if candidate is None else candidate.store_skip_end_token
        self._requests[request_id] = request
        self._allocations[request_id] = RequestAllocation(
            block_ids_by_group,
            block_hashes,
            store_skip_end,
        )
        if candidate is None:
            return

        expected_external_tokens = candidate.num_new_matched_tokens
        if allocated_external_tokens not in (0, expected_external_tokens):
            raise ValueError(
                f"Allocation confirmed {allocated_external_tokens} external tokens for request {request_id}; "
                f"expected either 0 or {expected_external_tokens}"
            )
        selected_for_load = expected_external_tokens > 0 and allocated_external_tokens == expected_external_tokens
        self._confirm_load_candidate(request_id, candidate, selected_for_load=selected_for_load)

    def _lookup_start_token(self, local_cached_tokens: int, query_end: int) -> int:
        return min(local_cached_tokens, query_end)

    def _confirm_load_candidate(
        self,
        request_id: str,
        candidate: LoadCandidate,
        *,
        selected_for_load: bool,
    ) -> None:
        raise NotImplementedError

    # ================================
    # Scheduler Step Publication
    # ================================

    def build_step(self, scheduler_output: SchedulerOutput) -> KVTransferStep:
        finished_request_ids = frozenset(scheduler_output.finished_req_ids)
        preempted_request_ids = frozenset(scheduler_output.preempted_req_ids or ())
        for request_id in finished_request_ids | preempted_request_ids:
            self._discard_request(request_id)

        load_commands: list[LoadCommand] = []
        source_pending_commands: list[RangeStoreCommand] = []
        source_ready_commands: list[CheckpointStoreCommand] = []

        for scheduled in scheduler_output.scheduled_new_reqs:
            load_command, store_command = self._plan_new_request(
                scheduled.req_id,
                _freeze_block_ids(scheduled.block_ids),
                scheduled.num_computed_tokens,
                scheduler_output.num_scheduled_tokens[scheduled.req_id],
            )
            _append_command(load_command, load_commands)
            _append_command(store_command, source_pending_commands)

        cached = scheduler_output.scheduled_cached_reqs
        resumed_request_ids = frozenset(cached.resumed_req_ids)
        for index, request_id in enumerate(cached.req_ids):
            block_ids = None if cached.new_block_ids[index] is None else _freeze_block_ids(cached.new_block_ids[index])
            if request_id in resumed_request_ids:
                load_command, store_command = self._plan_resumed_request(
                    request_id,
                    block_ids,
                    cached.num_computed_tokens[index],
                    scheduler_output.num_scheduled_tokens[request_id],
                )
            else:
                load_command, store_command = self._plan_running_request(
                    request_id,
                    block_ids,
                    cached.num_computed_tokens[index],
                    scheduler_output.num_scheduled_tokens[request_id],
                )
            _append_command(load_command, load_commands)
            _append_command(store_command, source_pending_commands)

        load_commands.extend(self._take_allocation_ready_loads())
        source_ready_commands.extend(self._plan_step_checkpoint_stores(scheduler_output.kv_connector_block_state))
        source_ready_commands.extend(self._finished_checkpoint_stores)
        self._finished_checkpoint_stores.clear()
        return KVTransferStep(
            LoadCommandBatch(tuple(load_commands)),
            StoreCommandBatch(tuple(source_pending_commands), tuple(source_ready_commands)),
        )

    def _plan_new_request(
        self,
        request_id: str,
        block_ids_by_group: tuple[tuple[int, ...], ...],
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> tuple[LoadCommand | None, RangeStoreCommand | None]:
        request, allocation = self._allocated_request(request_id)
        if block_ids_by_group != allocation.block_ids_by_group:
            raise ValueError(f"Scheduled NEW request {request_id} does not match its confirmed allocation")
        scheduled_end = num_computed_tokens + num_scheduled_tokens
        progress = RequestProgress(
            request_id,
            scheduled_end,
            block_ids_by_group,
            tuple(request.block_hashes),
            request.num_prompt_tokens,
            allocation.store_skip_end_token,
            min(scheduled_end, request.num_tokens),
        )
        self._request_progress[request_id] = progress
        load_command = self._take_scheduled_load(request_id)
        store_command = (
            None
            if load_command is not None and not self.store_with_scheduled_load
            else self._plan_range_store(progress)
        )
        return load_command, store_command

    def _plan_resumed_request(
        self,
        request_id: str,
        replacement_block_ids: tuple[tuple[int, ...], ...] | None,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> tuple[LoadCommand | None, RangeStoreCommand | None]:
        if replacement_block_ids is None:
            raise ValueError(f"Scheduled RESUMED request {request_id} has no replacement block table")
        request, allocation = self._allocated_request(request_id)
        if replacement_block_ids != allocation.block_ids_by_group:
            raise ValueError(f"Scheduled RESUMED request {request_id} does not match its confirmed allocation")
        scheduled_end = num_computed_tokens + num_scheduled_tokens
        progress = RequestProgress(
            request_id,
            scheduled_end,
            replacement_block_ids,
            tuple(request.block_hashes),
            request.num_tokens,
            allocation.store_skip_end_token,
            min(scheduled_end, request.num_tokens),
        )
        self._request_progress[request_id] = progress
        load_command = self._take_scheduled_load(request_id)
        store_command = (
            None
            if load_command is not None and not self.store_with_scheduled_load
            else self._plan_range_store(progress)
        )
        return load_command, store_command

    def _plan_running_request(
        self,
        request_id: str,
        new_block_ids: tuple[tuple[int, ...], ...] | None,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> tuple[LoadCommand | None, RangeStoreCommand | None]:
        request = self._requests.get(request_id)
        progress = self._request_progress.get(request_id)
        if request is None or progress is None:
            raise ValueError(f"Scheduled RUNNING request {request_id} has no Scheduler progress")

        scheduled_end = num_computed_tokens + num_scheduled_tokens
        progress = progress.advance(
            scheduled_end,
            new_block_ids,
            request.block_hashes,
            min(scheduled_end, request.num_tokens),
        )
        self._request_progress[request_id] = progress
        load_command = self._plan_running_load(progress, num_computed_tokens)
        is_decoding = num_computed_tokens >= progress.prefill_end_token
        store_command = None if is_decoding and not self._config.save_decode_cache else self._plan_range_store(progress)
        return load_command, store_command

    def _take_scheduled_load(self, request_id: str) -> LoadCommand | None:
        raise NotImplementedError

    def _take_allocation_ready_loads(self) -> list[LoadCommand]:
        raise NotImplementedError

    def _plan_running_load(self, progress: RequestProgress, num_computed_tokens: int) -> LoadCommand | None:
        del progress, num_computed_tokens
        return None

    def _make_load_command(self, request_id: str, candidate: LoadCandidate) -> LoadCommand | None:
        allocation = self._allocations.get(request_id)
        if allocation is None:
            raise ValueError(f"Request {request_id} is ready for Load without a confirmed allocation")
        if candidate.load_range.end_token <= candidate.load_range.start_token:
            return None
        return LoadCommand(
            request_id,
            candidate.load_range,
            allocation.block_ids_by_group,
            allocation.block_hashes,
            candidate.tail_key_boundaries,
        )

    # ================================
    # Store Publication and Ownership
    # ================================

    def _plan_range_store(self, progress: RequestProgress) -> RangeStoreCommand | None:
        if not self._config.store_enabled:
            return None
        transfer_end = self._resolve_transfer_end(progress.store_end_token, len(progress.block_hashes))
        if transfer_end <= progress.store_skip_end_token:
            return None
        if self._config.discard_partial_chunks:
            next_chunk_end = (
                cdiv(progress.store_skip_end_token + 1, self._config.cache_transfer_granularity)
                * self._config.cache_transfer_granularity
            )
            if transfer_end < next_chunk_end:
                return None

        command = RangeStoreCommand(
            progress.request_id,
            TokenRange(progress.store_skip_end_token, transfer_end),
            progress.block_ids_by_group,
            progress.block_hashes,
            progress.prefill_end_token,
        )
        leased_command = self._source_leases.acquire(command)
        self._request_progress[progress.request_id] = progress.with_store_skip_end(transfer_end)
        return leased_command

    def _plan_step_checkpoint_stores(self, block_state: Any) -> list[CheckpointStoreCommand]:
        if not self._config.store_enabled or block_state is None:
            return []

        commands = []
        for request_id, entries in block_state.boundary_state_offloads.items():
            progress = self._request_progress.get(request_id)
            if progress is None:
                continue
            entries_by_boundary: dict[int, list[tuple[int, int]]] = {}
            for group_id, block_id, boundary_token in entries:
                if (
                    group_id in self._config.transfer_group_ids
                    and group_id in self._config.align_state_group_ids
                    and block_id > 0
                    and 0 < boundary_token <= progress.request_token_len
                ):
                    entries_by_boundary.setdefault(boundary_token, []).append((group_id, block_id))
            for boundary_token, boundary_entries in sorted(entries_by_boundary.items()):
                command = CheckpointStoreCommand(
                    request_id,
                    progress.block_ids_by_group,
                    progress.block_hashes,
                    progress.store_skip_end_token,
                    tuple(
                        StateCheckpointSource(group_id, block_id, boundary_token)
                        for group_id, block_id in boundary_entries
                    ),
                )
                commands.append(self._source_leases.acquire(command))
        return commands

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        if not self._config.store_enabled or not partial_tail_offloads:
            return False
        progress = self._request_progress.get(request.request_id)
        if progress is None or not any(block_ids):
            return False
        boundaries = {boundary for _, _, boundary in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError("Finished state checkpoints must share one token boundary")
        boundary = next(iter(boundaries))
        if boundary <= 0 or boundary > progress.prefill_end_token:
            return False
        if (
            boundary % self._config.hash_block_size
            or boundary > len(request.block_hashes) * self._config.hash_block_size
        ):
            return False
        sources = tuple(
            StateCheckpointSource(group_id, block_id, boundary)
            for group_id, block_id, boundary in partial_tail_offloads
            if group_id in self._config.transfer_group_ids
            and group_id in self._config.align_state_group_ids
            and block_id > 0
        )
        if len(sources) != len(partial_tail_offloads):
            return False
        command = CheckpointStoreCommand(
            request.request_id,
            _freeze_block_ids(block_ids),
            tuple(request.block_hashes),
            progress.store_skip_end_token,
            sources,
        )
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

    # ================================
    # Request and Scheduler Lifecycle
    # ================================

    def _allocated_request(self, request_id: str) -> tuple[Request, RequestAllocation]:
        request = self._requests.get(request_id)
        allocation = self._allocations.get(request_id)
        if request is None or allocation is None:
            raise ValueError(f"Scheduled request {request_id} has not passed allocation confirmation")
        return request, allocation

    def _discard_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)
        self._allocations.pop(request_id, None)
        self._request_progress.pop(request_id, None)
        self._pending_lookups.pop(request_id, None)
        self._discard_confirmed_load(request_id)

    def _discard_confirmed_load(self, request_id: str) -> None:
        raise NotImplementedError

    def _resolve_transfer_end(self, target_token_len: int, num_block_hashes: int) -> int:
        publish_granularity = (
            self._config.cache_transfer_granularity
            if self._config.discard_partial_chunks
            else self._config.hash_block_size
        )
        hash_extent = num_block_hashes * self._config.hash_block_size
        available_end = min(target_token_len, hash_extent)
        return available_end // publish_granularity * publish_granularity

    def close(self) -> None:
        self._remote_lookup.close()
        pending_job_ids = self._source_leases.pending_job_ids()
        queued_checkpoints = len(self._finished_checkpoint_stores)
        if pending_job_ids or queued_checkpoints:
            raise RuntimeError(
                "Cannot close KV Pool Scheduler with pending Store ownership: "
                f"source lease job ids={pending_job_ids}, queued finished checkpoints={queued_checkpoints}"
            )


def _freeze_block_ids(block_ids_by_group: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(block_ids) for block_ids in block_ids_by_group)


def _append_command(command: CommandT | None, destination: list[CommandT]) -> None:
    if command is not None:
        destination.append(command)
