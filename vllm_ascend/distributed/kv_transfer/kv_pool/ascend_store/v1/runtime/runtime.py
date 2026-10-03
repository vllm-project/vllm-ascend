"""Apply bound KV rules at the business points selected by Runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeVar

import numpy as np
import torch
from vllm.logger import logger

from ...attention_fence import reset_attention_compute_start_gate
from ..backend import LayerwiseAccessKind
from ..coordinates import TokenRange
from ..program.spec.compilation import KVPoolCompilationSpec
from ..program.values.evidence import ChunkAvailability, GroupAvailability
from ..program.values.selection import KVSelection
from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommand,
)
from ..rules import KVPoolRules, RuleBinder
from ..rules.identity import BlockRows, KeyAxes
from .backend import BackendIO, GVABackendIO, KeyRangeBackendIO
from .backend.io import _batch_sources
from .batch import KVGroupBatch, KVTransferBatch
from .evidence import LoadCompletion, StoreCompletion, StoreEvidence, TransferEvidence
from .resources import KVPoolResources
from .result import LoadResult
from .timeline import StoreBatch
from .timeline.composition import KVPoolTimelineRuntime

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


@dataclass(slots=True)
class _KVPoolStepContext:
    """Adapt one upstream hook batch without owning cross-step executions."""

    step: KVTransferStep
    failed_request_ids: set[str] = field(default_factory=set)
    failed_block_ids: set[int] = field(default_factory=set)
    store_batch: KVTransferBatch | None = None
    store_submitted: bool = False


class KVPoolRuntime:
    """Join static rules, dynamic requests, Backend I/O and timeline state."""

    def __init__(
        self,
        spec: KVPoolCompilationSpec,
        rule_binder: RuleBinder,
        resources: KVPoolResources,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        topology = spec.topology
        timeline = KVPoolTimelineRuntime(spec.schedule, topology)
        backend_io: BackendIO
        layerwise_backend: GVABackendIO | KeyRangeBackendIO | None = None
        if spec.schedule.requires_layerwise_backend:
            access_kind = resources.backend_spec.layerwise_access
            if access_kind is None:
                raise ValueError("Layerwise timeline requires a session Backend")
            backend_type: type[GVABackendIO] | type[KeyRangeBackendIO] = (
                GVABackendIO if access_kind is LayerwiseAccessKind.GVA else KeyRangeBackendIO
            )
            layerwise_backend = backend_type(resources.backend, resources.backend_spec)
            backend_io = layerwise_backend
        else:
            backend_io = BackendIO(resources.backend, resources.backend_spec)

        self._spec = spec
        self._rule_binder = rule_binder
        self._rules: KVPoolRules | None = None
        self._resources = resources
        self._backend_io = backend_io
        self._timeline = timeline
        self._groups = {group.group_id: group for group in topology.transfer_groups}
        self._active_step: _KVPoolStepContext | None = None
        self._pending_load_request_ids: set[str] = set()
        self._pending_store_batch: StoreBatch | None = None
        self._released_store_job_ids: set[int] = set()
        self._store_error: Exception | None = None
        self._timeline.bind_resources(
            thread_initializer=backend_io.initialize_thread,
            source_ready_event_factory=source_ready_event_factory or (lambda: torch.npu.Event()),
            load_operation=self._execute_load,
            store_operation=self._execute_store,
            store_admission=self._admit_store,
            layerwise_backend=layerwise_backend,
            start_gate_factory=start_gate_factory,
        )

    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            registration = self._resources.bind_kv_caches(kv_caches)
            self._rules = self._rule_binder(**registration)
            self._backend_io.bind_rules(self._rules)
            self._timeline.start()
        except BaseException:
            self.close()
            raise

    def lookup(self, request: LookupRequest) -> LookupResult:
        rules = self._bound_rules
        if request.transfer_group_ids != rules.group_ids:
            raise ValueError(
                f"Lookup groups {request.transfer_group_ids} do not match configured groups {rules.group_ids}"
            )
        selection: KVSelection = rules.lookup_selection(request.block_hashes, request.query_range)
        availability = []
        for group in selection.groups:
            starts, counts, hashes = rules.lookup_chunks(
                group.group_id,
                request.query_range.end_token,
                request.block_hashes,
                start_token=request.query_range.start_token,
                mask=group.chunk_mask,
            )
            keys = rules.lookup_keys(group.group_id, hashes)
            try:
                observed = self._backend_io.exists([key for axis in keys for key in axis])
            except Exception as error:
                logger.error("Remote Lookup failed. type=%s, error=%s", type(error).__name__, error)
                return LookupResult(0)
            row_count = len(hashes)
            available = tuple(
                all(observed[axis_index * row_count + row_index] for axis_index in range(len(keys)))
                for row_index in range(row_count)
            )
            availability.append(
                GroupAvailability(
                    group.group_id,
                    tuple(
                        ChunkAvailability(TokenRange(int(start), int(start + count)), content_hash, hit)
                        for start, count, content_hash, hit in zip(starts, counts, hashes, available, strict=True)
                    ),
                )
            )
        reachable = rules.resolve_lookup(selection, availability)
        return LookupResult(reachable.end_token, reachable.tail_key_boundaries)

    def begin_step(self, step: KVTransferStep) -> None:
        self._raise_store_error()
        if self._active_step is not None:
            raise RuntimeError("Previous KV Pool step has not ended")
        context = _KVPoolStepContext(step)
        if self._timeline.store_enabled and step.store.commands:
            try:
                context.store_batch = self._build_store_batch(step.store.commands)
                self._timeline.prepare_store(context.store_batch)
                if step.store.all_sources_ready:
                    self._submit_store(context)
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
        method: Callable[Concatenate[KVPoolRuntime, _KVPoolStepContext, _Parameters], _Result],
    ) -> Callable[Concatenate[KVPoolRuntime, _Parameters], _Result]:
        @wraps(method)
        def guarded(self: KVPoolRuntime, *args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Result:
            if self._active_step is None:
                raise RuntimeError("KV Pool step has not begun")
            return method(self, self._active_step, *args, **kwargs)

        return guarded

    @_with_active_step
    def start_load(self, context: _KVPoolStepContext) -> None:
        batch = self._build_load_batch(context.step.load.commands)
        if self._timeline.collects_load_completions:
            overlapping = self._pending_load_request_ids.intersection(batch.request_ids)
            if overlapping:
                raise RuntimeError(f"Request already has a pending asynchronous Load: {sorted(overlapping)}")
            self._pending_load_request_ids.update(batch.request_ids)
        try:
            completions = self._timeline.submit_load(batch)
        except BaseException:
            self._pending_load_request_ids.difference_update(batch.request_ids)
            raise
        for completion in completions:
            self._record_load_completion(context, completion)
        self._raise_load_error(context)

    @_with_active_step
    def collect_load_result(self, context: _KVPoolStepContext) -> LoadResult:
        completions = self._timeline.collect_load()
        for completion in completions:
            if completion.request_id not in self._pending_load_request_ids:
                raise RuntimeError(f"Load completion has no pending execution: {completion.request_id}")
            self._pending_load_request_ids.remove(completion.request_id)
            self._record_load_completion(context, completion)
        result = LoadResult(
            frozenset(completion.request_id for completion in completions),
            frozenset(context.failed_request_ids),
            frozenset(context.failed_block_ids),
        )
        context.failed_request_ids.clear()
        context.failed_block_ids.clear()
        return result

    @_with_active_step
    def wait_for_layer_load(self, context: _KVPoolStepContext, layer_name: str) -> None:
        for completion in self._timeline.wait_for_load_layer(layer_name):
            self._record_load_completion(context, completion)
        self._raise_load_error(context)

    @_with_active_step
    def save_layer(self, _context: _KVPoolStepContext, layer_name: str) -> None:
        self._raise_store_error()
        try:
            self._timeline.submit_store_layer(layer_name)
        except Exception as error:
            self._store_error = error
            raise

    @_with_active_step
    def finish_step(self, context: _KVPoolStepContext) -> None:
        self._raise_store_error()
        if not self._timeline.store_enabled or context.store_batch is None or context.store_submitted:
            return
        try:
            self._submit_store(context)
            if self._timeline.fences_store_on_finish:
                self.fence_previous_store()
        except Exception as error:
            self._store_error = error
            raise

    def _submit_store(self, context: _KVPoolStepContext) -> None:
        if self._pending_store_batch is not None:
            raise RuntimeError("Previous Store invocation has not reached its fence")
        assert context.store_batch is not None
        self._pending_store_batch = self._timeline.finish_store(context.store_batch)
        context.store_submitted = True

    def fence_previous_store(self) -> tuple[StoreCompletion, ...]:
        self._raise_store_error()
        pending = self._pending_store_batch
        if not self._timeline.store_enabled or pending is None:
            return ()
        try:
            completions = self._timeline.wait_store(pending)
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
        try:
            if self._pending_store_batch is None:
                self._pending_store_batch = self._timeline.prepare_store_close()
            store_handoff_complete = True
            self.fence_previous_store()
        except BaseException as error:
            close_error = error
        try:
            self._timeline.close()
        except BaseException as error:
            if close_error is None:
                close_error = error
        finally:
            if store_handoff_complete and self._pending_store_batch is None:
                self._resources.close()
        if close_error is not None:
            raise close_error

    def _execute_load(
        self,
        batch: KVTransferBatch,
        layer_id: int | None,
    ) -> tuple[LoadCompletion, ...]:
        return self._backend_io.load_batch(batch, layer_id)

    def _admit_store(self, batch: KVTransferBatch) -> KVTransferBatch:
        rules = self._bound_rules
        selected_keys = batch.selected_keys()
        keys = tuple(dict.fromkeys(selected_keys))
        has_duplicates = len(keys) != len(selected_keys)
        if not rules.requires_store_observation and not has_duplicates:
            return batch
        admitted = rules.admit_store(self._backend_io.exists(list(keys))) if rules.requires_store_observation else None
        if admitted is None:
            accepted = set(keys)
        else:
            if not has_duplicates and all(admitted):
                return batch
            accepted = {key for key, include in zip(keys, admitted, strict=True) if include}
        return batch.select_keys(accepted, claim_once=has_duplicates)

    def _execute_store(
        self,
        batch: KVTransferBatch,
        source_ready_event: Any,
        layer_id: int | None,
    ) -> tuple[StoreCompletion, ...]:
        try:
            selected = self._admit_store(batch) if layer_id is None else batch
        except Exception as error:
            evidence = tuple(TransferEvidence(source, None, True) for source in _batch_sources(batch, layer_id))
            return _failed_store_completions(batch, evidence, error)
        if selected.empty:
            return self._backend_io.store_batch(selected, layer_id)
        try:
            source_ready_event.synchronize()
        except Exception as error:
            evidence = tuple(TransferEvidence(source, None, True) for source in _batch_sources(selected, layer_id))
            return _failed_store_completions(selected, evidence, error)
        return self._backend_io.store_batch(selected, layer_id)

    def _build_load_batch(self, commands: tuple[LoadCommand, ...]) -> KVTransferBatch:
        rules = self._bound_rules
        selections = tuple(rules.load_selection(command.block_hashes, command.load_range) for command in commands)
        rows_by_request: list[dict[int, BlockRows]] = []
        for command, selection in zip(commands, selections, strict=True):
            boundaries = {item.group_id: item.boundary_token for item in command.tail_key_boundaries}
            if len(boundaries) != len(command.tail_key_boundaries):
                raise ValueError("Load contains duplicate tail-key boundaries for one cache group")
            unknown = set(boundaries).difference(rules.group_ids)
            if unknown:
                raise ValueError(f"Load tail-key boundaries contain unknown cache groups {sorted(unknown)}")
            masks = {group.group_id: group.chunk_mask for group in selection.groups}
            rows_by_request.append(
                {
                    group_id: rules.load_rows(
                        group_id,
                        command.load_range.end_token,
                        command.block_hashes,
                        command.block_ids_by_group[group_id],
                        start_token=command.load_range.start_token,
                        mask=masks[group_id],
                        tail_boundary_token=boundaries.get(group_id),
                    )
                    for group_id in rules.group_ids
                }
            )
        return self._assemble_batch(
            tuple(command.request_id for command in commands),
            rows_by_request,
            rules.load_keys,
        )

    def _build_store_batch(self, commands: tuple[StoreCommand, ...]) -> KVTransferBatch:
        rules = self._bound_rules
        rows_by_request = []
        for command in commands:
            if isinstance(command, CheckpointStoreCommand):
                if rules.checkpoint_rows is None:
                    raise ValueError("State checkpoint Store requires an align-state cache group")
                boundaries = {source.boundary_token for source in command.sources}
                if len(boundaries) != 1:
                    raise ValueError("State checkpoints for one request must share one token boundary")
                boundary = next(iter(boundaries))
                source_blocks = {source.group_id: source.block_id for source in command.sources}
                rows = dict(
                    rules.checkpoint_rows(
                        boundary,
                        command.block_hashes,
                        source_blocks,
                        {group_id: command.block_ids_by_group[group_id] for group_id in rules.group_ids},
                        published_store_end_token=command.published_store_end_token,
                    )
                )
                rows_by_request.append(rows)
                continue
            assert isinstance(command, RangeStoreCommand)
            selection: KVSelection = rules.store_selection(
                command.block_hashes,
                command.store_range,
                command.num_prompt_tokens,
            )
            masks = {group.group_id: group.chunk_mask for group in selection.groups}
            rows_by_request.append(
                {
                    group_id: rules.store_rows(
                        group_id,
                        command.store_range.end_token,
                        command.block_hashes,
                        command.block_ids_by_group[group_id],
                        start_token=command.store_range.start_token,
                        mask=masks[group_id],
                    )
                    for group_id in rules.group_ids
                }
            )
        return self._assemble_batch(
            tuple(command.request_id for command in commands),
            rows_by_request,
            rules.store_keys,
            tuple(command.store_job_id for command in commands),
        )

    def _assemble_batch(
        self,
        request_ids: tuple[str, ...],
        rows_by_request: list[dict[int, BlockRows]],
        key_rule: Callable[[int, Sequence], KeyAxes],
        store_job_ids: tuple[int | None, ...] | None = None,
    ) -> KVTransferBatch:
        groups = []
        for group_id in self._bound_rules.group_ids:
            block_parts = []
            count_parts = []
            key_parts: list[KeyAxes] = []
            request_splits = [0]
            for rows_by_group in rows_by_request:
                rows = rows_by_group.get(group_id, _empty_rows())
                _, counts, hashes, block_ids = rows
                block_parts.append(block_ids)
                count_parts.append(counts)
                key_parts.append(key_rule(group_id, hashes))
                request_splits.append(request_splits[-1] + len(block_ids))
            key_axis_count = len(key_parts[0]) if key_parts else 0
            if any(len(keys) != key_axis_count for keys in key_parts):
                raise RuntimeError(f"Cache group {group_id} changed its key coordinate count at runtime")
            key_axes = tuple(
                tuple(key for keys in key_parts for key in keys[axis_index]) for axis_index in range(key_axis_count)
            )
            topology = self._groups[group_id]
            groups.append(
                KVGroupBatch(
                    group_id,
                    _concat_uint64(block_parts),
                    _concat_uint64(count_parts),
                    key_axes,
                    _readonly_indices(request_splits),
                    tuple(layer.physical_layer_id for layer in topology.layers),
                    self._bound_rules.object_size(group_id),
                )
            )
        return KVTransferBatch(request_ids, tuple(groups), store_job_ids)

    def _raise_store_error(self) -> None:
        if self._store_error is not None:
            raise RuntimeError("KVPoolRuntime cannot continue after a previous Store failure") from self._store_error

    def _record_load_completion(self, context: _KVPoolStepContext, completion: LoadCompletion) -> None:
        failed_block_ids = {item.source.block_id for item in completion.transfer_evidence if item.result_code != 0}
        if not failed_block_ids:
            return
        if len(self._bound_rules.group_ids) > 1:
            context.failed_request_ids.add(completion.request_id)
        else:
            context.failed_block_ids.update(failed_block_ids)

    def _raise_load_error(self, context: _KVPoolStepContext) -> None:
        if not context.failed_request_ids:
            return
        self._timeline.abort_load()
        raise RuntimeError(f"Hybrid KV Load failed for requests: {sorted(context.failed_request_ids)}")

    @property
    def _bound_rules(self) -> KVPoolRules:
        if self._rules is None:
            raise RuntimeError("KV Pool rules are unavailable before cache registration")
        return self._rules


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
    batch: KVTransferBatch,
    evidence: tuple[TransferEvidence, ...],
    error: Exception,
) -> tuple[StoreCompletion, ...]:
    by_request: list[list[TransferEvidence]] = [[] for _ in batch.request_ids]
    for item in evidence:
        by_request[item.source.request_index].append(item)
    job_ids = batch.store_job_ids or (None,) * len(batch.request_ids)
    return tuple(
        StoreCompletion(
            request_id,
            StoreEvidence(
                tuple(items),
                succeeded=not items,
                source_release_confirmed=True,
                error=error if items else None,
            ),
            store_job_id,
        )
        for request_id, store_job_id, items in zip(batch.request_ids, job_ids, by_request, strict=True)
    )


def _empty_rows() -> BlockRows:
    empty: np.ndarray = np.empty(0, dtype=np.uint64)
    empty.flags.writeable = False
    return empty, empty, (), empty


def _concat_uint64(parts: Sequence[np.ndarray]) -> np.ndarray:
    if not parts:
        result: np.ndarray = np.empty(0, dtype=np.uint64)
    elif len(parts) == 1:
        result = parts[0]
    else:
        result = np.concatenate(parts)
        result.flags.writeable = False
    return result


def _readonly_indices(values: Sequence[int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.intp)
    result.flags.writeable = False
    return result
