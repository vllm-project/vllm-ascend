"""Apply bound KV rules at the business points selected by Runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeAlias, TypeVar

import numpy as np
import torch
from vllm.logger import logger

from ...attention_fence import reset_attention_compute_start_gate
from ..backend import LayerwiseAccessKind
from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    StoreCommand,
)
from ..rules import KVPoolRules, RuleBinder
from ..rules.identity import BlockRows, KeyAxes, StoreCandidateRows
from ..timeline import StoreBatch
from ..timeline.schedule import KVPoolSchedule
from ..timeline.timeline import KVPoolTimeline
from ..topology import KVPoolTopology
from .backend import BackendIO, GVABackendIO, KeyRangeBackendIO
from .backend.io import _batch_sources
from .batch import (
    KVGroupBatch,
    KVTransferBatch,
    LayerStoreGroup,
    make_layer_store_group,
    make_layer_store_plan,
)
from .evidence import LayerStoreResult, LoadCompletion, StoreCompletion, StoreEvidence, TransferEvidence
from .resources import KVPoolResources
from .result import LoadResult

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")
_StoreCandidateRowsByGroup: TypeAlias = tuple[StoreCandidateRows, ...]
_StorePreparationResult: TypeAlias = tuple[KVTransferBatch | None, tuple[StoreCompletion, ...]]


@dataclass(slots=True)
class _KVPoolStepContext:
    """Adapt one upstream hook batch without owning cross-step executions."""

    step: KVTransferStep
    failed_request_ids: set[str] = field(default_factory=set)
    failed_block_ids: set[int] = field(default_factory=set)
    store_submitted: bool = False


@dataclass(frozen=True, slots=True)
class _StoreGroupCandidates:
    """One cache Group's list-backed rows and object-key coordinates."""

    group_id: int
    rows_by_request: tuple[StoreCandidateRows, ...]
    key_axes: KeyAxes

    @property
    def row_count(self) -> int:
        return sum(len(rows[0]) for rows in self.rows_by_request)

    @property
    def object_count(self) -> int:
        return len(self.key_axes) * self.row_count


@dataclass(frozen=True, slots=True)
class _StoreCandidates:
    """All pre-admission Store objects in canonical group/axis/row order."""

    groups: tuple[_StoreGroupCandidates, ...]


@dataclass(frozen=True, slots=True)
class _MaterializedStoreGroup:
    """One domain Group plus its optional Layerwise execution operands."""

    batch: KVGroupBatch
    layer_store: LayerStoreGroup | None


class KVPoolRuntime:
    """Join static rules, dynamic requests, Backend I/O and timeline state."""

    def __init__(
        self,
        topology: KVPoolTopology,
        schedule: KVPoolSchedule,
        rule_binder: RuleBinder,
        resources: KVPoolResources,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        timeline = KVPoolTimeline(schedule, topology)
        backend_io: BackendIO
        layerwise_backend: GVABackendIO | KeyRangeBackendIO | None = None
        if schedule.requires_layerwise_backend:
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
            bulk_store_operation=self._execute_bulk_store,
            layerwise_store_operation=self._execute_layerwise_store,
            layerwise_store_preparation=self._prepare_layerwise_store,
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
        masks = rules.lookup_selection(request.block_hashes, request.query_range)
        observations = []
        for group_id, mask in zip(rules.group_ids, masks, strict=True):
            starts, counts, hashes = rules.lookup_chunks(
                group_id,
                request.query_range.end_token,
                request.block_hashes,
                start_token=request.query_range.start_token,
                mask=mask,
            )
            keys = rules.lookup_keys(group_id, hashes)
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
            observations.append(((starts, counts, hashes), available))
        reachable = rules.resolve_lookup(request.query_range, request.block_hashes, observations)
        return LookupResult(reachable.end_token, reachable.tail_key_boundaries)

    def begin_step(self, step: KVTransferStep) -> None:
        self._raise_store_error()
        if self._active_step is not None:
            raise RuntimeError("Previous KV Pool step has not ended")
        context = _KVPoolStepContext(step)
        if self._timeline.store_enabled and step.store.commands:
            try:
                if self._timeline.prepares_store_by_layer:
                    self._timeline.prepare_store(step.store.commands)
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
        if not self._timeline.store_enabled or not context.step.store.commands or context.store_submitted:
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
        self._pending_store_batch = self._timeline.finish_store(context.step.store.commands)
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

    def _execute_load(self, batch: KVTransferBatch, layer_id: int | None) -> tuple[LoadCompletion, ...]:
        return self._backend_io.load_batch(batch, layer_id)

    def _execute_bulk_store(
        self, commands: tuple[StoreCommand, ...], source_ready_event: Any
    ) -> tuple[StoreCompletion, ...]:
        selected: KVTransferBatch | None = None
        try:
            selected = self._build_admitted_store_batch(commands)
            if selected is None:
                return _store_command_completions(commands)
            source_ready_event.synchronize()
        except Exception as error:
            if selected is None:
                return _store_command_completions(commands, error)
            evidence = tuple(TransferEvidence(source, None, True) for source in _batch_sources(selected, None))
            return _failed_store_completions(selected, evidence, error)
        return self._backend_io.store_batch(selected)

    def _build_admitted_store_batch(
        self,
        commands: tuple[StoreCommand, ...],
        *,
        prepare_layerwise: bool = False,
    ) -> KVTransferBatch | None:
        candidates = self._build_store_candidates(commands)
        candidate_keys = _store_candidate_keys(candidates)
        accepted, claim_once = self._admitted_store_keys(candidate_keys)
        if accepted is not None and not accepted:
            return None
        selected_objects = (
            None
            if accepted is None
            else _select_store_candidate_objects(candidate_keys, accepted, claim_once=claim_once)
        )
        return self._materialize_store_candidates(
            commands,
            candidates,
            selected_objects,
            prepare_layerwise=prepare_layerwise,
        )

    def _materialize_store_candidates(
        self,
        commands: tuple[StoreCommand, ...],
        candidates: _StoreCandidates,
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
        return KVTransferBatch(
            tuple(command.request_id for command in commands),
            tuple(groups),
            tuple(command.store_job_id for command in commands),
            selected_keys,
            make_layer_store_plan(tuple(layer_store_groups)) if prepare_layerwise else None,
        )

    def _materialize_store_group(
        self,
        candidates: _StoreGroupCandidates,
        selected_objects: tuple[bool, ...] | None,
        *,
        prepare_layerwise: bool,
    ) -> _MaterializedStoreGroup:
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
            token_counts, block_ids, request_splits, selected_axis = _compact_single_axis_rows(
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
            materialized_selection = None if all(compact_selection) else _readonly_bool(compact_selection)
            if materialized_selection is not None:
                selected_object_indices = np.flatnonzero(materialized_selection)
            selected_keys = tuple(
                key
                for axis, selection in zip(key_axes, _split_selection(compact_selection, len(key_axes)), strict=True)
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
        group = KVGroupBatch(
            candidates.group_id,
            _readonly_uint64(block_ids),
            _readonly_uint64(token_counts),
            key_axes,
            _readonly_indices(request_splits),
            tuple(layer.physical_layer_id for layer in topology.layers),
            self._bound_rules.object_size(candidates.group_id),
            materialized_selection,
            selected_keys,
        )
        layer_store = (
            make_layer_store_group(group, selected_object_indices) if prepare_layerwise and selected_keys else None
        )
        return _MaterializedStoreGroup(group, layer_store)

    def _admitted_store_keys(self, selected_keys: tuple[str, ...]) -> tuple[set[str] | None, bool]:
        if not selected_keys:
            return set(), False
        rules = self._bound_rules
        keys = tuple(dict.fromkeys(selected_keys))
        has_duplicates = len(keys) != len(selected_keys)
        if not rules.requires_store_observation:
            return (set(keys), True) if has_duplicates else (None, False)
        admitted = rules.admit_store(self._backend_io.exists(list(keys)))
        if not has_duplicates and all(admitted):
            return None, False
        return {key for key, include in zip(keys, admitted, strict=True) if include}, has_duplicates

    def _prepare_layerwise_store(self, commands: tuple[StoreCommand, ...]) -> _StorePreparationResult:
        try:
            selected = self._build_admitted_store_batch(commands, prepare_layerwise=True)
        except Exception as error:
            return None, _store_command_completions(commands, error)
        if selected is None:
            return None, _store_command_completions(commands)
        return selected, ()

    def _execute_layerwise_store(self, source_ready_event: Any, layer_id: int) -> LayerStoreResult:
        try:
            source_ready_event.synchronize()
        except Exception as error:
            return LayerStoreResult(None, True, error)
        layerwise_backend = self._backend_io
        assert isinstance(layerwise_backend, (GVABackendIO, KeyRangeBackendIO))
        return layerwise_backend.store_layer(layer_id)

    def _build_load_batch(self, commands: tuple[LoadCommand, ...]) -> KVTransferBatch:
        rules = self._bound_rules
        masks_by_request = tuple(rules.load_selection(command.block_hashes, command.load_range) for command in commands)
        rows_by_request: list[dict[int, BlockRows]] = []
        for command, masks in zip(commands, masks_by_request, strict=True):
            boundaries = {item.group_id: item.boundary_token for item in command.tail_key_boundaries}
            if len(boundaries) != len(command.tail_key_boundaries):
                raise ValueError("Load contains duplicate tail-key boundaries for one cache group")
            unknown = set(boundaries).difference(rules.group_ids)
            if unknown:
                raise ValueError(f"Load tail-key boundaries contain unknown cache groups {sorted(unknown)}")
            rows_by_request.append(
                {
                    group_id: rules.load_rows(
                        group_id,
                        command.load_range.end_token,
                        command.block_hashes,
                        command.block_ids_by_group[group_id],
                        start_token=command.load_range.start_token,
                        mask=mask,
                        tail_boundary_token=boundaries.get(group_id),
                    )
                    for group_id, mask in zip(rules.group_ids, masks, strict=True)
                }
            )
        return self._assemble_batch(tuple(command.request_id for command in commands), rows_by_request, rules.load_keys)

    def _build_store_candidates(self, commands: tuple[StoreCommand, ...]) -> _StoreCandidates:
        rules = self._bound_rules
        rows_by_group: list[list[StoreCandidateRows]] = [[] for _group_id in rules.group_ids]
        keys_by_group: list[list[KeyAxes]] = [[] for _group_id in rules.group_ids]
        for command in commands:
            for group_index, (group_id, rows) in enumerate(
                zip(rules.group_ids, self._store_candidate_rows(command), strict=True)
            ):
                rows_by_group[group_index].append(rows)
                keys_by_group[group_index].append(rules.store_keys(group_id, rows[1]))
        return _StoreCandidates(
            tuple(
                _StoreGroupCandidates(group_id, tuple(rows), _merge_key_axes(group_id, keys_by_group[group_index]))
                for group_index, (group_id, rows) in enumerate(zip(rules.group_ids, rows_by_group, strict=True))
            )
        )

    def _store_candidate_rows(self, command: StoreCommand) -> _StoreCandidateRowsByGroup:
        rules = self._bound_rules
        if isinstance(command, CheckpointStoreCommand):
            if rules.checkpoint_rows is None:
                raise ValueError("State checkpoint Store requires an align-state cache group")
            boundaries = {source.boundary_token for source in command.sources}
            if len(boundaries) != 1:
                raise ValueError("State checkpoints for one request must share one token boundary")
            boundary = next(iter(boundaries))
            source_blocks = {source.group_id: source.block_id for source in command.sources}
            rows_by_group = dict(
                rules.checkpoint_rows(
                    boundary,
                    command.block_hashes,
                    source_blocks,
                    {group_id: command.block_ids_by_group[group_id] for group_id in rules.group_ids},
                    published_store_end_token=command.published_store_end_token,
                )
            )
            return tuple(rows_by_group.get(group_id, _empty_candidate_rows()) for group_id in rules.group_ids)

        masks = rules.store_selection(command.block_hashes, command.store_range, command.num_prompt_tokens)
        return tuple(
            rules.store_candidate_rows(
                group_id,
                command.store_range.end_token,
                command.block_hashes,
                command.block_ids_by_group[group_id],
                start_token=command.store_range.start_token,
                mask=mask,
            )
            for group_id, mask in zip(rules.group_ids, masks, strict=True)
        )

    def _assemble_batch(
        self,
        request_ids: tuple[str, ...],
        rows_by_request: list[dict[int, BlockRows]],
        key_rule: Callable[[int, Sequence], KeyAxes],
    ) -> KVTransferBatch:
        groups = []
        for group_id in self._bound_rules.group_ids:
            block_parts = []
            count_parts = []
            request_splits = [0]
            for rows_by_group in rows_by_request:
                rows = rows_by_group.get(group_id, _empty_rows())
                counts, _, block_ids = rows
                block_parts.append(block_ids)
                count_parts.append(counts)
                request_splits.append(request_splits[-1] + len(block_ids))
            key_axes = _merge_key_axes(
                group_id,
                [
                    key_rule(group_id, rows_by_group.get(group_id, _empty_rows())[1])
                    for rows_by_group in rows_by_request
                ],
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
        return KVTransferBatch(request_ids, tuple(groups))

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


def _store_candidate_keys(candidates: _StoreCandidates) -> tuple[str, ...]:
    return tuple(key for group in candidates.groups for axis in group.key_axes for key in axis)


def _select_store_candidate_objects(
    candidate_keys: tuple[str, ...], accepted: set[str], *, claim_once: bool
) -> tuple[bool, ...]:
    if not claim_once:
        return tuple(key in accepted for key in candidate_keys)
    claimed = set()
    selected = []
    for key in candidate_keys:
        include = key in accepted and key not in claimed
        selected.append(include)
        if include:
            claimed.add(key)
    return tuple(selected)


def _merge_key_axes(group_id: int, key_parts: list[KeyAxes]) -> KeyAxes:
    key_axis_count = len(key_parts[0]) if key_parts else 0
    if any(len(keys) != key_axis_count for keys in key_parts):
        raise RuntimeError(f"Cache group {group_id} changed its key coordinate count at runtime")
    return tuple(tuple(key for keys in key_parts for key in keys[axis_index]) for axis_index in range(key_axis_count))


def _split_selection(selection: tuple[bool, ...], axis_count: int) -> tuple[tuple[bool, ...], ...]:
    if axis_count == 0:
        return ()
    if len(selection) % axis_count:
        raise RuntimeError("Store selection does not form aligned key axes")
    row_count = len(selection) // axis_count
    return tuple(selection[index * row_count : (index + 1) * row_count] for index in range(axis_count))


def _compact_single_axis_rows(
    rows_by_request: tuple[StoreCandidateRows, ...],
    key_axis: tuple[str, ...],
    selected_rows: tuple[bool, ...],
) -> tuple[list[int], list[int], list[int], tuple[str, ...]]:
    token_counts: list[int] = []
    block_ids: list[int] = []
    selected_keys: list[str] = []
    request_splits = [0]
    row_offset = 0
    for counts, _hashes, ids in rows_by_request:
        request_row_count = len(counts)
        selection = selected_rows[row_offset : row_offset + request_row_count]
        keys = key_axis[row_offset : row_offset + request_row_count]
        for count, block_id, key, keep in zip(counts, ids, keys, selection, strict=True):
            if keep:
                token_counts.append(count)
                block_ids.append(block_id)
                selected_keys.append(key)
        request_splits.append(len(block_ids))
        row_offset += request_row_count
    return token_counts, block_ids, request_splits, tuple(selected_keys)


def _empty_candidate_rows() -> StoreCandidateRows:
    return [], [], []


def _empty_rows() -> BlockRows:
    empty: np.ndarray = np.empty(0, dtype=np.uint64)
    empty.flags.writeable = False
    return empty, (), empty


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


def _readonly_uint64(values: Sequence[int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.uint64)
    result.flags.writeable = False
    return result


def _readonly_bool(values: Sequence[bool]) -> np.ndarray:
    result = np.asarray(values, dtype=np.bool_)
    result.flags.writeable = False
    return result
