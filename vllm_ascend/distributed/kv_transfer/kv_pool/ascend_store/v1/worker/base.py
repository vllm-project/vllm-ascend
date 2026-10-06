"""Own common KV Pool request lifecycle, evidence, fences, and release."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeAlias, TypeVar

import numpy as np
import torch
from vllm.logger import logger

from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import KVTransferStep, LoadCommand, StoreCommand
from ..timeline.asynchronous_load import AsynchronousLoadTimeline
from ..timeline.asynchronous_store import AsynchronousStoreTimeline
from ..timeline.layerwise_load import LayerwiseLoadTimeline
from ..timeline.layerwise_store import LayerwiseStoreTimeline
from ..timeline.store_batch import StoreBatch
from ..timeline.synchronous_load import SynchronousLoadTimeline
from ..topology import KVPoolTopology
from .io import BackendIO
from .resources import KVPoolResources
from .transfer.batch import KeyAxes, KVGroupBatch, KVTransferBatch, make_layer_store_group, make_layer_store_plan
from .transfer.evidence import LoadCompletion, StoreCompletion, StoreEvidence, TransferEvidence
from .transfer.result import LoadFailureLocation, LoadResult
from .transfer.state import (
    KVPoolStepContext,
    MaterializedStoreGroup,
    StoreCandidates,
    StoreGroupCandidates,
    compact_single_axis_rows,
    readonly_bool,
    readonly_indices,
    readonly_uint64,
    split_selection,
)

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")
_LoadTimeline: TypeAlias = SynchronousLoadTimeline | AsynchronousLoadTimeline | LayerwiseLoadTimeline
_StoreTimeline: TypeAlias = AsynchronousStoreTimeline | LayerwiseStoreTimeline


class KVPoolWorker(ABC):
    """Own request lifecycle, evidence aggregation, and safe resource release."""

    def __init__(
        self,
        topology: KVPoolTopology,
        resources: KVPoolResources,
        backend_io: BackendIO,
        load_timeline: _LoadTimeline,
        store_timeline: _StoreTimeline | None,
        *,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._topology = topology
        self._resources = resources
        self._backend_io = backend_io
        self._load_timeline = load_timeline
        self._store_timeline = store_timeline
        self._groups = {group.group_id: group for group in topology.transfer_groups}
        self._active_step: KVPoolStepContext | None = None
        self._pending_load_request_ids: set[str] = set()
        self._pending_store_batch: StoreBatch | None = None
        self._released_store_job_ids: set[int] = set()
        self._store_error: Exception | None = None
        self._timelines_started = False
        self._source_ready_event_factory = source_ready_event_factory or (lambda: torch.npu.Event())

    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            registration = self._resources.bind_kv_caches(kv_caches)
            self._bind_projection(registration)
            if self._store_timeline is not None:
                self._store_timeline.start()
            self._load_timeline.start()
            self._timelines_started = True
        except BaseException:
            try:
                self.close()
            except BaseException:
                logger.exception("Failed to close KV Pool resources after cache registration failed")
            raise

    @abstractmethod
    def _bind_projection(self, registration: dict[str, Any]) -> None:
        """Bind the route-specific immutable projection after memory registration."""

    @abstractmethod
    def lookup(self, request: LookupRequest) -> LookupResult:
        """Resolve the readable remote prefix for one request."""

    def begin_step(self, step: KVTransferStep) -> None:
        self._raise_store_error()
        if self._active_step is not None:
            raise RuntimeError("Previous KV Pool step has not ended")
        context = KVPoolStepContext(step)
        if self._store_timeline is not None and step.store.commands:
            try:
                self._prepare_step_store(step.store.commands)
                if step.store.all_sources_ready:
                    self._submit_step_store(context, fence=False)
            except Exception as error:
                self._store_error = error
                raise
        self._active_step = context

    def end_step(self) -> None:
        if self._active_step is None:
            raise RuntimeError("KV Pool step has not begun")
        self._active_step = None

    @staticmethod
    def _with_active_step(
        method: Callable[Concatenate[KVPoolWorker, KVPoolStepContext, _Parameters], _Result],
    ) -> Callable[Concatenate[KVPoolWorker, _Parameters], _Result]:
        @wraps(method)
        def guarded(self: KVPoolWorker, *args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Result:
            if self._active_step is None:
                raise RuntimeError("KV Pool step has not begun")
            return method(self, self._active_step, *args, **kwargs)

        return guarded

    @_with_active_step
    def start_load(self, context: KVPoolStepContext) -> None:
        batch = self._build_load_batch(context.step.load.commands)
        self._before_load_submit(batch)
        try:
            completions = self._load_timeline.submit(batch)
        except BaseException:
            self._load_submit_failed(batch)
            raise

        for completion in completions:
            self._record_load_completion(context, completion)
        self._raise_load_error(context)

    @_with_active_step
    def collect_load_result(self, context: KVPoolStepContext) -> LoadResult:
        completions: Sequence[LoadCompletion] = self._load_timeline.collect()

        for completion in completions:
            self._accept_load_completion(completion)
            self._record_load_completion(context, completion)
        result = LoadResult(
            frozenset(completion.request_id for completion in completions),
            frozenset(context.failed_request_ids),
            frozenset(context.failed_block_ids),
            tuple(
                dict.fromkeys(
                    LoadFailureLocation(source.group_id, source.block_id) for source in context.failed_sources
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
        self._raise_store_error()
        try:
            self._submit_store_layer(layer_name)
        except Exception as error:
            self._store_error = error
            raise

    @_with_active_step
    def finish_step(self, context: KVPoolStepContext) -> None:
        self._raise_store_error()
        if self._store_timeline is None or not context.step.store.commands or context.store_submitted:
            return
        try:
            self._submit_step_store(context, fence=True)
        except Exception as error:
            self._store_error = error
            raise

    @abstractmethod
    def _submit_step_store(self, context: KVPoolStepContext, *, fence: bool) -> None:
        """Publish one route-specific Store invocation into the common fence."""

    def _prepare_step_store(self, commands: tuple[StoreCommand, ...]) -> None:
        """Prepare route-specific Store state before source readiness."""

        del commands

    def _prepare_store_close(self) -> StoreBatch | None:
        return None

    def _submit_store_layer(self, layer_name: str) -> None:
        """Submit one layer when the selected route is layerwise."""

        del layer_name

    def _wait_for_layer_load(self, layer_name: str) -> tuple[LoadCompletion, ...]:
        return ()

    def _before_load_submit(self, batch: KVTransferBatch) -> None:
        """Reserve route-specific Load state before Timeline submission."""

        del batch

    def _load_submit_failed(self, batch: KVTransferBatch) -> None:
        """Roll back route-specific Load state after submission failure."""

        del batch

    def _accept_load_completion(self, completion: LoadCompletion) -> None:
        """Consume route-specific pending state for one completion."""

        del completion

    def fence_previous_store(self) -> tuple[StoreCompletion, ...]:
        self._raise_store_error()
        pending = self._pending_store_batch
        if pending is None:
            return ()
        try:
            if self._store_timeline is None:
                raise RuntimeError("Store completion exists without a Store timeline")
            completions = self._store_timeline.wait(pending)
        except Exception as error:
            self._store_error = error
            raise
        for completion in completions:
            if completion.evidence.source_release_confirmed and completion.store_job_id is not None:
                self._released_store_job_ids.add(completion.store_job_id)
        if all(completion.evidence.source_release_confirmed for completion in completions):
            self._pending_store_batch = None
        for completion in completions:
            try:
                _validate_store_completion(completion)
            except Exception as error:
                self._store_error = error
                raise
        return completions

    def take_released_store_job_ids(self) -> set[int]:
        released = self._released_store_job_ids
        self._released_store_job_ids = set()
        return released

    def close(self) -> None:
        close_error: BaseException | None = None
        store_handoff_complete = False
        if not self._timelines_started and self._pending_store_batch is None:
            store_handoff_complete = True
        else:
            try:
                if self._pending_store_batch is None:
                    self._pending_store_batch = self._prepare_store_close()
                store_handoff_complete = True
                self.fence_previous_store()
            except BaseException as error:
                close_error = error

        store_timeline_stopped = True
        if self._store_timeline is not None:
            try:
                self._store_timeline.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
                else:
                    logger.exception("Failed to close Store Timeline after an earlier close failure")
            store_timeline_stopped = self._store_timeline.stopped

        load_timeline_stopped = False
        try:
            self._load_timeline.close()
        except BaseException as error:
            if close_error is None:
                close_error = error
            else:
                logger.exception("Failed to close Load Timeline after an earlier close failure")
        load_timeline_stopped = self._load_timeline.stopped

        if (
            store_handoff_complete
            and self._pending_store_batch is None
            and store_timeline_stopped
            and load_timeline_stopped
        ):
            try:
                self._resources.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
                else:
                    logger.exception("Failed to close KV Pool resources after Timeline close failed")
        if close_error is not None:
            raise close_error

    def _record_source_ready(self) -> Any:
        source_ready_event = self._source_ready_event_factory()
        source_ready_event.record()
        return source_ready_event

    def _materialize_store_candidates(
        self,
        commands: tuple[StoreCommand, ...],
        candidates: StoreCandidates,
        selected_objects: tuple[bool, ...] | None,
        *,
        prepare_layerwise: bool = False,
    ) -> KVTransferBatch:
        groups = []
        layer_store_groups = []
        object_offset = 0
        for group_candidates in candidates.groups:
            next_offset = object_offset + group_candidates.object_count
            group_selection = None if selected_objects is None else selected_objects[object_offset:next_offset]
            materialized = self._materialize_store_group(
                group_candidates,
                group_selection,
                prepare_layerwise=prepare_layerwise,
            )
            groups.append(materialized.batch)
            if materialized.layer_store is not None:
                layer_store_groups.append(materialized.layer_store)
            object_offset = next_offset
        if selected_objects is not None and object_offset != len(selected_objects):
            raise RuntimeError("Store candidate selection does not match the candidate object count")

        selected_keys = tuple(key for group in groups for key in group.selected_keys())
        batch = KVTransferBatch(
            tuple(command.request_id for command in commands),
            tuple(groups),
            tuple(command.store_job_id for command in commands),
            selected_keys,
        )
        if prepare_layerwise:
            batch = replace(
                batch,
                layer_store_plan=make_layer_store_plan(tuple(layer_store_groups), batch),
            )
        return batch

    def _materialize_store_group(
        self,
        candidates: StoreGroupCandidates,
        selected_objects: tuple[bool, ...] | None,
        *,
        prepare_layerwise: bool,
    ) -> MaterializedStoreGroup:
        row_count = candidates.row_count
        for counts, hashes, block_ids in candidates.rows_by_request:
            if len(hashes) != len(counts) or len(block_ids) != len(counts):
                raise RuntimeError(f"Cache group {candidates.group_id} produced misaligned Store candidate rows")
        if any(len(axis) != row_count for axis in candidates.key_axes):
            raise RuntimeError(f"Cache group {candidates.group_id} produced misaligned Store candidate keys")
        if selected_objects is not None and len(selected_objects) != candidates.object_count:
            raise RuntimeError(
                f"Cache group {candidates.group_id} Store selection does not match its candidate objects"
            )

        key_axes: KeyAxes
        selected_object_indices = None
        if selected_objects is not None and len(candidates.key_axes) == 1:
            token_counts, block_ids, request_splits, selected_axis = compact_single_axis_rows(
                candidates.rows_by_request,
                candidates.key_axes[0],
                selected_objects,
            )
            key_axes = (selected_axis,)
            materialized_selection = None
            selected_keys = selected_axis
        elif selected_objects is None:
            keep_rows: tuple[bool, ...] | None = None
            key_axes = candidates.key_axes
            materialized_selection = None
            selected_keys = tuple(key for axis in key_axes for key in axis)
        else:
            selections_by_axis = tuple(
                selected_objects[axis_index * row_count : (axis_index + 1) * row_count]
                for axis_index in range(len(candidates.key_axes))
            )
            keep_rows = tuple(any(axis[row_index] for axis in selections_by_axis) for row_index in range(row_count))
            key_axes = tuple(
                tuple(key for key, keep in zip(axis, keep_rows, strict=True) if keep) for axis in candidates.key_axes
            )
            compact_selection = tuple(
                include for axis in selections_by_axis for include, keep in zip(axis, keep_rows, strict=True) if keep
            )
            materialized_selection = None if all(compact_selection) else readonly_bool(compact_selection)
            if materialized_selection is not None:
                selected_object_indices = np.flatnonzero(materialized_selection)
            selected_keys = tuple(
                key
                for axis, selection in zip(key_axes, split_selection(compact_selection, len(key_axes)), strict=True)
                for key, include in zip(axis, selection, strict=True)
                if include
            )

        if selected_objects is None or len(candidates.key_axes) != 1:
            token_counts = []
            block_ids = []
            request_splits = [0]
            row_offset = 0
            for counts, _hashes, ids in candidates.rows_by_request:
                request_row_count = len(counts)
                if keep_rows is None:
                    token_counts.extend(counts)
                    block_ids.extend(ids)
                else:
                    request_selection = keep_rows[row_offset : row_offset + request_row_count]
                    token_counts.extend(count for count, keep in zip(counts, request_selection, strict=True) if keep)
                    block_ids.extend(block_id for block_id, keep in zip(ids, request_selection, strict=True) if keep)
                request_splits.append(len(block_ids))
                row_offset += request_row_count

        topology = self._groups[candidates.group_id]
        object_size, physical_layer_ids_by_axis = self._store_group_layout(candidates.group_id)
        group = KVGroupBatch(
            candidates.group_id,
            readonly_uint64(block_ids),
            readonly_uint64(token_counts),
            key_axes,
            readonly_indices(request_splits),
            tuple(layer.physical_layer_id for layer in topology.layers),
            object_size,
            materialized_selection,
            selected_keys,
            physical_layer_ids_by_axis,
        )
        layer_store = (
            make_layer_store_group(group, selected_object_indices) if prepare_layerwise and selected_keys else None
        )
        return MaterializedStoreGroup(group, layer_store)

    def _admitted_store_keys(self, selected_keys: tuple[str, ...]) -> tuple[set[str] | None, bool]:
        if not selected_keys:
            return set(), False
        keys = tuple(dict.fromkeys(selected_keys))
        has_duplicates = len(keys) != len(selected_keys)
        requires_store_observation = self._resources.backend_spec.requires_exists_before_put
        if not requires_store_observation:
            return (set(keys), True) if has_duplicates else (None, False)
        observed = self._backend_io.exists(list(keys))
        admitted = tuple(not present for present in observed)
        if not has_duplicates and all(admitted):
            return None, False
        return {key for key, include in zip(keys, admitted, strict=True) if include}, has_duplicates

    @abstractmethod
    def _build_load_batch(self, commands: tuple[LoadCommand, ...]) -> KVTransferBatch:
        """Build the immutable Backend batch for the selected route."""

    @abstractmethod
    def _build_store_candidates(self, commands: tuple[StoreCommand, ...]) -> StoreCandidates:
        """Build request rows and keys before Store admission."""

    def _raise_store_error(self) -> None:
        if self._store_error is not None:
            raise RuntimeError("KVPoolWorker cannot continue after a previous Store failure") from self._store_error

    def _record_load_completion(self, context: KVPoolStepContext, completion: LoadCompletion) -> None:
        failed_sources = [item.source for item in completion.transfer_evidence if item.result_code != 0]
        if not failed_sources:
            return
        context.failed_sources.extend(failed_sources)
        if self._transfer_group_count > 1:
            context.failed_request_ids.add(completion.request_id)
        else:
            context.failed_block_ids.update(source.block_id for source in failed_sources)

    def _raise_load_error(self, context: KVPoolStepContext) -> None:
        if not context.failed_request_ids:
            return
        self._load_timeline.abort()
        failed_locations = sorted({(source.group_id, source.block_id) for source in context.failed_sources})
        raise RuntimeError(
            f"Hybrid KV Load failed for requests {sorted(context.failed_request_ids)}; "
            f"cache group/block failures {failed_locations}"
        )

    @abstractmethod
    def _store_group_layout(self, group_id: int) -> tuple[int, tuple[tuple[int, ...], ...] | None]:
        """Return object size and optional key-axis layer provenance."""

    @property
    @abstractmethod
    def _transfer_group_count(self) -> int:
        """Return the number of cache groups represented in one request result."""


def _validate_store_completion(completion: StoreCompletion) -> None:
    evidence = completion.evidence
    if evidence.error is not None:
        raise RuntimeError(f"Store failed for request {completion.request_id}") from evidence.error
    if not evidence.succeeded:
        raise RuntimeError(
            f"Store success was not confirmed for request {completion.request_id}; "
            f"result codes {[item.result_code for item in evidence.transfer_evidence]}"
        )
    if not evidence.source_release_confirmed:
        raise RuntimeError(f"Store source release is unknown for request {completion.request_id}")


def _failed_store_completions(
    batch: KVTransferBatch, evidence: tuple[TransferEvidence, ...], error: Exception
) -> tuple[StoreCompletion, ...]:
    by_request: list[list[TransferEvidence]] = [[] for _ in batch.request_ids]
    for item in evidence:
        by_request[item.source.request_index].append(item)
    job_ids = batch.store_job_ids or (None,) * len(batch.request_ids)
    return tuple(
        StoreCompletion(
            request_id,
            StoreEvidence(
                tuple(items), succeeded=not items, source_release_confirmed=True, error=error if items else None
            ),
            store_job_id,
        )
        for request_id, store_job_id, items in zip(batch.request_ids, job_ids, by_request, strict=True)
    )


def _store_command_completions(
    commands: tuple[StoreCommand, ...], error: Exception | None = None
) -> tuple[StoreCompletion, ...]:
    evidence = StoreEvidence((), error is None, True, error)
    return tuple(StoreCompletion(command.request_id, evidence, command.store_job_id) for command in commands)
