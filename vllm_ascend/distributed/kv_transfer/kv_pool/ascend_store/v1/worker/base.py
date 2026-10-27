"""Own common KV Pool request lifecycle, evidence, fences, and release."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Concatenate, Generic, ParamSpec, TypeVar

import torch
from vllm.logger import logger

from ..infrastructure.configuration.topology import KVPoolTopology
from ..infrastructure.execution.executor import Executor
from ..infrastructure.mapping import BulkMapping, LayerwiseMapping
from ..infrastructure.protocol.messages import KVTransferStep, LoadCommand, LookupRequest, LookupResult, StoreCommand
from ..infrastructure.reachability import HybridReachability, UnitaryReachability
from ..infrastructure.resources import KVPoolResources
from .transfer.batch import (
    KeysByPartition,
    KVGroupBatch,
    KVTransferBatch,
    StoreGroupTransferCandidates,
    StoreTransferCandidates,
    TransferSource,
)
from .transfer.completion import LoadCompletion, StoreCompletion
from .transfer.result import LoadFailureLocation, LoadResult
from .transfer.rows import (
    compact_single_partition_records,
    concat_uint64,
    fine_lookup_chunk_ranges,
    load_transfer_records,
    lookup_chunk_ranges,
    merge_keys_by_partition,
    readonly_bool,
    readonly_indices,
    readonly_uint64,
    split_selection,
)

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")
_WorkerT = TypeVar("_WorkerT", bound="KVPoolWorker")
_MappingT = TypeVar("_MappingT", BulkMapping, LayerwiseMapping)


@dataclass(slots=True)
class KVPoolStepContext:
    """Adapt one upstream hook batch without owning cross-step executions."""

    step: KVTransferStep
    failed_request_ids: set[str] = field(default_factory=set)
    failed_block_ids: set[int] = field(default_factory=set)
    failed_sources: list[TransferSource] = field(default_factory=list)
    store_submitted: bool = False


@dataclass(frozen=True, slots=True)
class PendingStore:
    """Associate one submitted Store result with its requests and source leases."""

    future: Future[tuple[StoreCompletion, ...]]
    request_ids: frozenset[str]
    store_job_ids: frozenset[int]


class KVPoolWorker(ABC, Generic[_MappingT]):
    """Own request lifecycle, evidence aggregation, and safe resource release."""

    def __init__(
        self,
        topology: KVPoolTopology,
        resources: KVPoolResources,
        executors: Sequence[Executor],
        *,
        store_enabled: bool,
        reachability: UnitaryReachability | HybridReachability,
        kv_cache_layout: str = "NHD",
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._topology = topology
        self._reachability = reachability
        self._kv_cache_layout = kv_cache_layout
        self._mappings: dict[int, _MappingT] = {}
        self._state_group_ids = frozenset(
            group.group_id for group in topology.transfer_groups if group.uses_align_state
        )
        self._fine_grained_lookup = any(
            group.uses_align_state and group.block_size > topology.hash_block_size for group in topology.transfer_groups
        )
        self._resources = resources
        self._backend = resources.backend
        resources.bind_executors(executors)
        self._store_enabled = store_enabled
        self._groups = {group.group_id: group for group in topology.transfer_groups}
        self._active_step: KVPoolStepContext | None = None
        self._pending_store_batches: deque[PendingStore] = deque()
        self._pending_store_batch: PendingStore | None = None
        self._latest_store_batch_by_request: dict[str, PendingStore] = {}
        self._released_store_job_ids: set[int] = set()
        # Only source-safety and execution faults block later steps. Completed
        # cache write failures are logged at the Store fence.
        self._store_fatal_error: Exception | None = None
        self._execution_started = False
        self._source_ready_event_factory = source_ready_event_factory or (lambda: torch.npu.Event())

    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            registration = self._resources.prepare_kv_caches(kv_caches)
            self._bind_mappings(registration)
            self._resources.register_kv_buffers()
            self._resources.start_executors()
            self._execution_started = True
        except BaseException:
            try:
                self.close()
            except BaseException:
                logger.exception("Failed to close KV Pool resources after cache registration failed")
            raise

    @abstractmethod
    def _bind_mappings(self, registration: dict[str, Any]) -> None: ...

    def _exists(self, keys: list[str]) -> tuple[bool, ...]:
        return self._backend.exists(keys) if keys else ()

    def lookup(self, request: LookupRequest) -> LookupResult:
        if not self._execution_started:
            raise RuntimeError("KV caches must be bound before Lookup")
        if request.transfer_group_ids != self._topology.transfer_group_ids:
            raise ValueError(
                f"Lookup groups {request.transfer_group_ids} do not match configured groups "
                f"{self._topology.transfer_group_ids}"
            )
        masks = self._reachability.select_for_lookup(request.block_hashes, request.end_token)
        observations = []
        chunk_rows = fine_lookup_chunk_ranges if self._fine_grained_lookup else lookup_chunk_ranges
        for (group_id, mapping), mask in zip(self._mappings.items(), masks, strict=True):
            starts, counts, hashes = chunk_rows(
                request.end_token,
                request.block_hashes,
                block_size=mapping.local_layout.group_block_size,
                hash_block_size=mapping.local_layout.hash_block_size,
                start_token=request.start_token,
                mask=mask,
            )
            keys = self._mappings[group_id].key_namer.lookup_keys(hashes)
            try:
                observed = self._exists([key for axis in keys for key in axis])
            except Exception as error:
                logger.error("Remote Lookup failed. type=%s, error=%s", type(error).__name__, error)
                return LookupResult(0)
            transfer_record_count = len(hashes)
            available = tuple(
                all(
                    observed[partition_index * transfer_record_count + transfer_record_index]
                    for partition_index in range(len(keys))
                )
                for transfer_record_index in range(transfer_record_count)
            )
            observations.append(((starts, counts, hashes), available))
        reachable = self._reachability.resolve_available_end(
            request.start_token, request.end_token, request.block_hashes, observations
        )
        return LookupResult(reachable.end_token, reachable.tail_key_boundaries)

    def _build_load_batch(self, commands: tuple[LoadCommand, ...]) -> KVTransferBatch:
        transfer_records_by_request = []
        for command in commands:
            masks = self._reachability.select_for_load(command.block_hashes, command.end_token)
            boundaries = {item.group_id: item.boundary_token for item in command.tail_key_boundaries}
            if len(boundaries) != len(command.tail_key_boundaries):
                raise ValueError("Load contains duplicate tail-key boundaries for one cache group")
            unknown = set(boundaries).difference(self._groups)
            if unknown:
                raise ValueError(f"Load tail-key boundaries contain unknown cache groups {sorted(unknown)}")
            transfer_records_by_request.append(
                {
                    group_id: load_transfer_records(
                        command.end_token,
                        command.block_hashes,
                        command.block_ids_by_group[group_id],
                        block_size=mapping.local_layout.group_block_size,
                        hash_block_size=mapping.local_layout.hash_block_size,
                        start_token=command.start_token,
                        tail_boundary_token=boundaries.get(group_id),
                        mask=mask,
                        minimum_block_id=1 if group_id in self._state_group_ids else 0,
                    )
                    for (group_id, mapping), mask in zip(self._mappings.items(), masks, strict=True)
                }
            )
        groups = []
        for group_id, mapping in self._mappings.items():
            build_keys = self._mappings[group_id].key_namer.keys
            rows = [request_rows[group_id] for request_rows in transfer_records_by_request]
            splits = [0]
            for counts, hashes, local_block_ids in rows:
                splits.append(splits[-1] + len(local_block_ids))
            remote_object_size_bytes, physical_layer_ids_by_partition = self._group_object_layout(group_id)
            groups.append(
                KVGroupBatch(
                    group_id,
                    concat_uint64([row[2] for row in rows]),
                    merge_keys_by_partition(group_id, [build_keys(row[1]) for row in rows]),
                    readonly_indices(splits),
                    mapping.local_layout.physical_layer_ids,
                    remote_object_size_bytes,
                    physical_layer_ids_by_partition=physical_layer_ids_by_partition,
                )
            )
        return KVTransferBatch(tuple(command.request_id for command in commands), tuple(groups))

    def begin_step(self, step: KVTransferStep) -> None:
        if not self._execution_started:
            raise RuntimeError("KV caches must be bound before a transfer step")
        self._raise_store_fatal_error()
        if self._active_step is not None:
            raise RuntimeError("Previous KV Pool step has not ended")
        context = KVPoolStepContext(step)
        if self._store_enabled and step.store.commands:
            try:
                self._prepare_step_store(step.store.commands)
                if step.store.all_sources_ready:
                    self._submit_step_store(context, fence=False)
            except Exception as error:
                self._store_fatal_error = error
                raise
        self._active_step = context

    def end_step(self) -> None:
        if self._active_step is None:
            raise RuntimeError("KV Pool step has not begun")
        self._active_step = None

    @staticmethod
    def _with_active_step(
        method: Callable[Concatenate[_WorkerT, KVPoolStepContext, _Parameters], _Result],
    ) -> Callable[Concatenate[_WorkerT, _Parameters], _Result]:
        @wraps(method)
        def guarded(self: _WorkerT, *args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Result:
            if self._active_step is None:
                raise RuntimeError("KV Pool step has not begun")
            return method(self, self._active_step, *args, **kwargs)

        return guarded

    @_with_active_step
    def start_load(self, context: KVPoolStepContext) -> None:
        if not context.step.load_commands:
            return
        batch = self._build_load_batch(context.step.load_commands)
        completions = self._submit_load(batch)
        for completion in completions:
            self._record_load_completion(context, completion)
        self._raise_load_error(context)

    @_with_active_step
    def collect_load_result(self, context: KVPoolStepContext) -> LoadResult:
        completions = self._collect_load_completions()

        for completion in completions:
            self._accept_load_completion(completion)
            self._record_load_completion(context, completion)
        result = LoadResult(
            frozenset(completion.request_id for completion in completions),
            frozenset(context.failed_request_ids),
            frozenset(context.failed_block_ids),
            tuple(
                dict.fromkeys(
                    LoadFailureLocation(source.group_id, source.local_block_id) for source in context.failed_sources
                )
            ),
        )
        context.failed_request_ids.clear()
        context.failed_block_ids.clear()
        context.failed_sources.clear()
        return result

    @_with_active_step
    def wait_for_layer_load(self, context: KVPoolStepContext, layer_name: str) -> None:
        for completion in self._wait_for_layer_load(layer_name):
            self._record_load_completion(context, completion)
        self._raise_load_error(context)

    @_with_active_step
    def save_layer(self, _context: KVPoolStepContext, layer_name: str) -> None:
        self._raise_store_fatal_error()
        try:
            self._submit_store_layer(layer_name)
        except Exception as error:
            self._store_fatal_error = error
            raise

    @_with_active_step
    def finish_step(self, context: KVPoolStepContext) -> None:
        self._raise_store_fatal_error()
        if not self._store_enabled or not context.step.store.commands or context.store_submitted:
            return
        try:
            self._submit_step_store(context, fence=True)
        except Exception as error:
            self._store_fatal_error = error
            raise

    @abstractmethod
    def _submit_step_store(self, context: KVPoolStepContext, *, fence: bool) -> None:
        """Publish one route-specific Store invocation into the common fence."""

    def _prepare_step_store(self, commands: tuple[StoreCommand, ...]) -> None:
        """Prepare route-specific Store state before source readiness."""

        del commands

    def _prepare_store_close(self) -> PendingStore | None:
        return None

    def _submit_store_layer(self, layer_name: str) -> None:
        """Submit one layer when the selected route is layerwise."""

        del layer_name

    def _wait_for_layer_load(self, layer_name: str) -> tuple[LoadCompletion, ...]:
        return ()

    @abstractmethod
    def _submit_load(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        """Run or submit Load, returning only results available in the submitting call."""

    @abstractmethod
    def _collect_load_completions(self) -> Sequence[LoadCompletion]:
        """Consume route-specific Load completions at the result boundary."""

    def _abort_load(self) -> None:
        """Finish route-specific Load state after a fatal business result."""

        return None

    def _accept_load_completion(self, completion: LoadCompletion) -> None:
        """Consume route-specific pending state for one completion."""

        del completion

    def _track_store_batch(self, batch: PendingStore) -> None:
        if self._pending_store_batch is None:
            self._pending_store_batch = batch
        self._pending_store_batches.append(batch)
        for request_id in batch.request_ids:
            self._latest_store_batch_by_request[request_id] = batch

    def fence_previous_store(self, released_request_ids: frozenset[str] | None = None) -> tuple[StoreCompletion, ...]:
        """Poll completed Stores, fence released requests, or explicitly drain all with no selection.

        FIFO means waiting for a request's latest batch also completes its earlier
        batches. Like production, a normal step need not wait for sources owned
        by requests whose blocks are not being released.
        """
        self._raise_store_fatal_error()
        if self._pending_store_batch is None:
            return ()
        if not self._store_enabled:
            raise RuntimeError("Store completion exists while Store is disabled")
        required_batch_ids = (
            set()
            if released_request_ids is None
            else {
                id(self._latest_store_batch_by_request[request_id])
                for request_id in released_request_ids
                if request_id in self._latest_store_batch_by_request
            }
        )
        collected: list[StoreCompletion] = []
        try:
            while self._pending_store_batches:
                pending = self._pending_store_batches[0]
                completions = self._collect_store_completions(
                    pending, wait=released_request_ids is None or bool(required_batch_ids)
                )
                if completions is None:
                    break
                self._accept_store_completions(pending, completions)
                collected.extend(completions)
                self._pending_store_batches.popleft()
                self._pending_store_batch = self._pending_store_batches[0] if self._pending_store_batches else None
                required_batch_ids.discard(id(pending))
                for request_id in pending.request_ids:
                    if self._latest_store_batch_by_request.get(request_id) is pending:
                        del self._latest_store_batch_by_request[request_id]
        except Exception as error:
            self._store_fatal_error = error
            raise
        return tuple(collected)

    @abstractmethod
    def _collect_store_completions(self, batch: PendingStore, *, wait: bool) -> tuple[StoreCompletion, ...] | None:
        """Read one Store result; readiness alone does not confirm source release."""

    def _accept_store_completions(self, batch: PendingStore, completions: tuple[StoreCompletion, ...]) -> None:
        for completion in completions:
            if completion.evidence.source_release_confirmed and completion.store_job_id is not None:
                self._released_store_job_ids.add(completion.store_job_id)
        for completion in completions:
            evidence = completion.evidence
            if evidence.error is not None or not evidence.succeeded:
                logger.error(
                    "KV cache Store failed for request %s (job %s); result codes=%s, error=%s",
                    completion.request_id,
                    completion.store_job_id,
                    [item.result_code for item in evidence.transfer_evidence],
                    evidence.error,
                )
            if not evidence.source_release_confirmed:
                raise RuntimeError(
                    f"Store source release is unknown for request {completion.request_id}"
                ) from evidence.error
        observed_job_ids = {completion.store_job_id for completion in completions}
        missing_job_ids = batch.store_job_ids.difference(observed_job_ids)
        if missing_job_ids:
            raise RuntimeError(f"Store source release is missing for jobs {sorted(missing_job_ids)}")

    def take_released_store_job_ids(self) -> set[int]:
        released = self._released_store_job_ids
        self._released_store_job_ids = set()
        return released

    def close(self) -> None:
        close_error: BaseException | None = None
        store_handoff_complete = False
        if not self._execution_started and self._pending_store_batch is None:
            store_handoff_complete = True
        else:
            try:
                if not self._pending_store_batches:
                    closing_batch = self._prepare_store_close()
                    if closing_batch is not None:
                        self._track_store_batch(closing_batch)
                store_handoff_complete = True
                self.fence_previous_store()
            except BaseException as error:
                close_error = error

        try:
            self._resources.stop_executors()
        except BaseException as error:
            if close_error is None:
                close_error = error
            else:
                logger.exception("Failed to stop executors after an earlier close failure")

        executors_stopped = self._resources.executors_stopped
        if executors_stopped:
            try:
                self._check_execution_results()
            except BaseException as error:
                if close_error is None:
                    close_error = error
                else:
                    logger.exception("Failed to check execution results after an earlier close failure")

        if store_handoff_complete and self._pending_store_batch is None and executors_stopped:
            try:
                self._resources.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
                else:
                    logger.exception("Failed to close KV Pool resources after execution close failed")
        if close_error is not None:
            raise close_error

    def _check_execution_results(self) -> None:
        """Observe execution errors after draining, without consuming pending Load notifications."""

        return None

    def _record_source_ready(self) -> Any:
        source_ready_event = self._source_ready_event_factory()
        source_ready_event.record()
        return source_ready_event

    def _materialize_store_candidates(
        self,
        commands: tuple[StoreCommand, ...],
        candidates: StoreTransferCandidates,
        transfer_candidate_mask: tuple[bool, ...] | None,
    ) -> KVTransferBatch:
        groups = []
        object_offset = 0
        for group_candidates in candidates.groups:
            next_offset = object_offset + group_candidates.transfer_candidate_count
            group_selection = (
                None if transfer_candidate_mask is None else transfer_candidate_mask[object_offset:next_offset]
            )
            groups.append(self._materialize_store_group(group_candidates, group_selection))
            object_offset = next_offset
        if transfer_candidate_mask is not None and object_offset != len(transfer_candidate_mask):
            raise RuntimeError("Store candidate selection does not match the candidate object count")

        selected_keys = tuple(key for group in groups for key in group.selected_keys())
        return KVTransferBatch(
            tuple(command.request_id for command in commands),
            tuple(groups),
            tuple(command.store_job_id for command in commands),
            selected_keys,
        )

    def _materialize_store_group(
        self,
        candidates: StoreGroupTransferCandidates,
        transfer_candidate_mask: tuple[bool, ...] | None,
    ) -> KVGroupBatch:
        transfer_record_count = candidates.transfer_record_count
        for counts, hashes, local_block_ids in candidates.transfer_records_by_request:
            if len(hashes) != len(counts) or len(local_block_ids) != len(counts):
                raise RuntimeError(f"Cache group {candidates.group_id} produced misaligned Store candidate rows")
        if any(len(axis) != transfer_record_count for axis in candidates.keys_by_partition):
            raise RuntimeError(f"Cache group {candidates.group_id} produced misaligned Store candidate keys")
        if transfer_candidate_mask is not None and len(transfer_candidate_mask) != candidates.transfer_candidate_count:
            raise RuntimeError(
                f"Cache group {candidates.group_id} Store selection does not match its candidate objects"
            )

        keys_by_partition: KeysByPartition
        if transfer_candidate_mask is not None and len(candidates.keys_by_partition) == 1:
            local_block_ids, request_transfer_record_boundaries, selected_axis = compact_single_partition_records(
                candidates.transfer_records_by_request,
                candidates.keys_by_partition[0],
                transfer_candidate_mask,
            )
            keys_by_partition = (selected_axis,)
            materialized_selection = None
            selected_keys = selected_axis
        elif transfer_candidate_mask is None:
            keep_rows: tuple[bool, ...] | None = None
            keys_by_partition = candidates.keys_by_partition
            materialized_selection = None
            selected_keys = tuple(key for axis in keys_by_partition for key in axis)
        else:
            selections_by_axis = tuple(
                transfer_candidate_mask[
                    partition_index * transfer_record_count : (partition_index + 1) * transfer_record_count
                ]
                for partition_index in range(len(candidates.keys_by_partition))
            )
            keep_rows = tuple(
                any(axis[transfer_record_index] for axis in selections_by_axis)
                for transfer_record_index in range(transfer_record_count)
            )
            keys_by_partition = tuple(
                tuple(key for key, keep in zip(axis, keep_rows, strict=True) if keep)
                for axis in candidates.keys_by_partition
            )
            compact_selection = tuple(
                include for axis in selections_by_axis for include, keep in zip(axis, keep_rows, strict=True) if keep
            )
            materialized_selection = None if all(compact_selection) else readonly_bool(compact_selection)
            selected_keys = tuple(
                key
                for axis, selection in zip(
                    keys_by_partition, split_selection(compact_selection, len(keys_by_partition)), strict=True
                )
                for key, include in zip(axis, selection, strict=True)
                if include
            )

        if transfer_candidate_mask is None or len(candidates.keys_by_partition) != 1:
            local_block_ids = []
            request_transfer_record_boundaries = [0]
            transfer_record_start = 0
            for counts, _hashes, ids in candidates.transfer_records_by_request:
                request_transfer_record_count = len(counts)
                if keep_rows is None:
                    local_block_ids.extend(ids)
                else:
                    request_selection = keep_rows[
                        transfer_record_start : transfer_record_start + request_transfer_record_count
                    ]
                    local_block_ids.extend(
                        block_id for block_id, keep in zip(ids, request_selection, strict=True) if keep
                    )
                request_transfer_record_boundaries.append(len(local_block_ids))
                transfer_record_start += request_transfer_record_count

        remote_object_size_bytes, physical_layer_ids_by_partition = self._group_object_layout(candidates.group_id)
        return KVGroupBatch(
            candidates.group_id,
            readonly_uint64(local_block_ids),
            keys_by_partition,
            readonly_indices(request_transfer_record_boundaries),
            self._mappings[candidates.group_id].local_layout.physical_layer_ids,
            remote_object_size_bytes,
            materialized_selection,
            selected_keys,
            physical_layer_ids_by_partition,
        )

    def _admitted_store_keys(self, selected_keys: tuple[str, ...]) -> tuple[set[str] | None, bool]:
        if not selected_keys:
            return set(), False
        keys = tuple(dict.fromkeys(selected_keys))
        has_duplicates = len(keys) != len(selected_keys)
        requires_store_observation = self._resources.backend_spec.requires_exists_before_put
        if not requires_store_observation:
            return (set(keys), True) if has_duplicates else (None, False)
        observed = self._exists(list(keys))
        admitted = tuple(not present for present in observed)
        if not has_duplicates and all(admitted):
            return None, False
        return {key for key, include in zip(keys, admitted, strict=True) if include}, has_duplicates

    @abstractmethod
    def _build_store_candidates(self, commands: tuple[StoreCommand, ...]) -> StoreTransferCandidates:
        """Build request rows and keys before Store admission."""

    def _raise_store_fatal_error(self) -> None:
        if self._store_fatal_error is not None:
            raise RuntimeError("KVPoolWorker cannot continue after a fatal Store failure") from self._store_fatal_error

    def _record_load_completion(self, context: KVPoolStepContext, completion: LoadCompletion) -> None:
        if not completion.failures:
            return
        failed_sources = [item.source for item in completion.failures]
        context.failed_sources.extend(failed_sources)
        if self._transfer_group_count > 1:
            context.failed_request_ids.add(completion.request_id)
        else:
            context.failed_block_ids.update(source.local_block_id for source in failed_sources)

    def _raise_load_error(self, context: KVPoolStepContext) -> None:
        if not context.failed_request_ids:
            return
        self._abort_load()
        failed_locations = sorted({(source.group_id, source.local_block_id) for source in context.failed_sources})
        raise RuntimeError(
            f"Hybrid KV Load failed for requests {sorted(context.failed_request_ids)}; "
            f"cache group/block failures {failed_locations}"
        )

    def _group_object_layout(self, group_id: int) -> tuple[int, tuple[tuple[int, ...], ...]]:
        local = self._mappings[group_id].local_layout
        return local.local_group_block_bytes, local.physical_layer_ids_by_partition

    @property
    def _transfer_group_count(self) -> int:
        return len(self._groups)
