"""Shared Worker route for layer-restricted Backend sessions."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, CancelledError, Future
from dataclasses import dataclass, field
from itertools import repeat
from typing import Any, cast

import numpy as np
from vllm.logger import logger
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from ....attention_fence import reset_attention_compute_start_gate
from ...backend import ObjectSessions
from ...infrastructure.execution.executor import Executor
from ...infrastructure.execution.session import Session
from ...infrastructure.mapping import LayerwiseMapping
from ...infrastructure.reachability import HybridReachability, UnitaryReachability
from ...infrastructure.resources import KVPoolResources
from ...protocol.transfer import CheckpointStoreCommand, StoreCommand
from ...topology import KVPoolTopology
from ..base import KVPoolStepContext, KVPoolWorker, PendingStore
from ..transfer.batch import (
    KeyAxes,
    KVTransferBatch,
    StoreCandidates,
    StoreGroupCandidates,
    TransferSource,
    select_store_candidate_objects,
    store_candidate_keys,
)
from ..transfer.completion import (
    LoadCompletion,
    StoreCompletion,
    StoreEvidence,
    TransferEvidence,
    _failed_load_completions,
    _store_command_completions,
)
from ..transfer.rows import (
    StoreCandidateRows,
    empty_candidate_rows,
    merge_key_axes,
    store_candidate_rows,
)
from .transfer import LayerStoreResult, LayerTransferPlan


@dataclass(slots=True)
class _StoreState:
    """One Store's inputs and accumulated outcomes, retained by its queued tasks.

    Only the serial Store executor mutates this state. failure_codes contains
    failures only; an absent key is not yet a successful publication.
    """

    commands: tuple[StoreCommand, ...]
    batch: KVTransferBatch | None = None
    opened_keys: tuple[str, ...] = ()
    keys_by_layer: dict[int, list[str]] = field(default_factory=dict)
    failure_codes: dict[str, int | None] = field(default_factory=dict)
    unsafe_keys: set[str] = field(default_factory=set)
    errors_by_key: dict[str, Exception] = field(default_factory=dict)


class LayerwiseWorker(KVPoolWorker):
    """Own request and layer scheduling; Backend owns the remote object lifecycle."""

    def __init__(
        self,
        topology: KVPoolTopology,
        resources: KVPoolResources,
        backend_sessions: ObjectSessions,
        *,
        reachability: UnitaryReachability | HybridReachability,
        kv_cache_layout: str = "NHD",
        store_enabled: bool = True,
        layerwise_prefetch_layers: int = 2,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._layer_mappings: dict[int, LayerwiseMapping] = {}
        self._load_plan: LayerTransferPlan | None = None
        self._store_plan: LayerTransferPlan | None = None
        self._backend_sessions = backend_sessions
        resources.bind_backend_sessions(backend_sessions)
        self._layerwise_store_leader = _is_store_leader(topology)
        self._start_gate_factory = start_gate_factory
        self._prefetch_layers = layerwise_prefetch_layers
        self._layer_ids_by_name = compile_layer_ids_by_name(topology)
        self._session: Session[tuple[LoadCompletion, ...]] | None = None
        self._store_state: _StoreState | None = None
        self._store_opening: Future[None] | None = None
        self._load_batch: KVTransferBatch | None = None
        self._load_initial_completions: tuple[LoadCompletion, ...] = ()
        self._attention_store_layers: frozenset[int] = frozenset()
        self._load_executor = Executor("KVPoolLayerwiseLoadExecutor", resources.backend.set_device)
        self._store_executor = (
            Executor("KVPoolLayerwiseStoreExecutor", resources.backend.set_device) if store_enabled else None
        )
        super().__init__(
            topology,
            resources,
            executors=tuple(
                executor for executor in (self._store_executor, self._load_executor) if executor is not None
            ),
            store_enabled=store_enabled,
            reachability=reachability,
            kv_cache_layout=kv_cache_layout,
            source_ready_event_factory=source_ready_event_factory,
        )

    def _bind_mappings(self, registration: dict[str, Any]) -> None:
        for group_id in self._groups:
            mapping = self._create_mapping(group_id, registration)
            self._mappings[group_id] = mapping
            self._layer_mappings[group_id] = LayerwiseMapping(mapping)

    @abstractmethod
    def _prepare_load_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]: ...

    @abstractmethod
    def _prepare_store_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]: ...

    @abstractmethod
    def _copy_load_layer(self, layer_id: int) -> tuple[LoadCompletion, ...]: ...

    @abstractmethod
    def _copy_store_layer(self, source_ready_event: Any, layer_id: int) -> LayerStoreResult: ...

    def _prepared_load_plan(self) -> LayerTransferPlan:
        if self._load_plan is None:
            raise RuntimeError("Layerwise Load plan was not prepared")
        return self._load_plan

    def _prepared_store_plan(self) -> LayerTransferPlan:
        if self._store_plan is None:
            raise RuntimeError("Layerwise Store plan was not prepared")
        return self._store_plan

    def _submit_load(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        object_sizes = collect_object_sizes(batch)
        if not object_sizes:
            return ()
        self._load_batch = batch
        self._load_initial_completions = ()
        try:
            order = self._load_executor.submit(self._open_load, batch, object_sizes).result()
            if order:
                self._get_session().load_order = order
        except (BrokenExecutor, CancelledError):
            raise
        except Exception:
            logger.exception("Layerwise Load preparation failed for requests %s", batch.request_ids)
            self._abort_load()
            return _failed_load_completions(batch, None)
        return self._load_initial_completions

    def _open_load(self, batch: KVTransferBatch, object_sizes: dict[str, int]) -> tuple[int, ...]:
        self._load_plan = None
        result_codes = self._backend_sessions.start_load(list(object_sizes), list(object_sizes.values()))
        opened_keys = set()
        failed_codes = {}
        for key, code in zip(object_sizes, result_codes, strict=True):
            if code == 0:
                opened_keys.add(key)
            else:
                failed_codes[key] = code
        self._load_initial_completions = _load_session_failures(batch, failed_codes) if failed_codes else ()
        if not opened_keys:
            return ()
        selected = batch if not failed_codes else batch.select_keys(opened_keys)
        keys_by_layer = self._prepare_load_layers(selected)
        return tuple(sorted(layer_id for layer_id, keys in keys_by_layer.items() if keys))

    def _finish_load(self) -> None:
        self._load_plan = None
        self._backend_sessions.finish_load()

    def _collect_load_completions(self) -> tuple[LoadCompletion, ...]:
        if self._session is not None and (pending := self._session.pending_load_layers):
            self._abort_load()
            raise RuntimeError(f"Layerwise Load did not reach physical layers {list(pending)}")
        return ()

    def _abort_load(self, *, wait: bool = True) -> None:
        if self._session is not None or self._backend_sessions.has_load_sessions:
            finish = self._finish_load if self._backend_sessions.has_load_sessions else None
            self._get_session().abort_load(finish, wait=wait)

    def _collect_store_completions(self, batch: PendingStore, *, wait: bool) -> tuple[StoreCompletion, ...] | None:
        return batch.future.result() if wait or batch.future.done() else None

    def _get_session(self) -> Session[tuple[LoadCompletion, ...]]:
        if self._session is None:
            store_submit = self._store_executor.submit if self._store_executor is not None else None
            self._session = Session(
                self._load_executor.submit, store_submit, self._prefetch_layers, self._start_gate_factory
            )
        return self._session

    def _build_store_candidates(self, commands: tuple[StoreCommand, ...]) -> StoreCandidates:
        rows_by_group: list[list[StoreCandidateRows]] = [[] for _group_id in self._topology.transfer_group_ids]
        keys_by_group: list[list[KeyAxes]] = [[] for _group_id in self._topology.transfer_group_ids]
        for command in commands:
            for group_index, (group_id, rows) in enumerate(
                zip(self._topology.transfer_group_ids, self._store_candidate_rows(command), strict=True)
            ):
                rows_by_group[group_index].append(rows)
                keys_by_group[group_index].append(self._mappings[group_id].keys(rows[1]))
        return StoreCandidates(
            tuple(
                StoreGroupCandidates(group_id, tuple(rows), merge_key_axes(group_id, keys_by_group[group_index]))
                for group_index, (group_id, rows) in enumerate(
                    zip(self._topology.transfer_group_ids, rows_by_group, strict=True)
                )
            )
        )

    def _store_candidate_rows(self, command: StoreCommand) -> tuple[StoreCandidateRows, ...]:
        if isinstance(command, CheckpointStoreCommand):
            raise ValueError("Layerwise transfer does not support state checkpoint Store")

        if not self._layerwise_store_leader:
            return tuple(empty_candidate_rows() for _group_id in self._topology.transfer_group_ids)
        masks = self._reachability.select_for_store(
            command.block_hashes,
            command.store_range,
            command.num_prompt_tokens,
        )
        return tuple(
            store_candidate_rows(
                command.store_range.end_token,
                command.block_hashes,
                command.block_ids_by_group[group_id],
                block_size=self._mappings[group_id].group_block_size,
                hash_block_size=self._mappings[group_id].hash_block_size,
                start_token=command.store_range.start_token,
                mask=mask,
            )
            for group_id, mask in zip(self._topology.transfer_group_ids, masks, strict=True)
        )

    def _prepare_step_store(self, commands: tuple[StoreCommand, ...]) -> None:
        # Worker has one Store plan; a new Store waits, an empty decode does not.
        if self._pending_store_batch is not None:
            self.fence_previous_store()
        state = _StoreState(commands)
        self._store_state = state
        assert self._store_executor is not None
        # Presence checks run without a step Session or Backend leases and overlap model execution.
        self._store_opening = self._store_executor.submit(self._open_store, state)

    def _open_store(self, state: _StoreState) -> None:
        candidates = self._build_store_candidates(state.commands)
        candidate_keys = store_candidate_keys(candidates)
        accepted, claim_once = self._admitted_store_keys(candidate_keys)
        if accepted is not None and not accepted:
            return
        selected_objects = (
            None
            if accepted is None
            else select_store_candidate_objects(candidate_keys, accepted, claim_once=claim_once)
        )
        batch = self._materialize_store_candidates(state.commands, candidates, selected_objects)
        state.batch = batch
        # Admission has already chosen one source owner per key, including duplicates across requests.
        keys, object_sizes = [], []
        for group in batch.groups:
            group_keys = group.selected_keys()
            keys.extend(group_keys)
            object_sizes.extend([group.object_size] * len(group_keys))
        self._store_plan = None
        try:
            result_codes = self._backend_sessions.start_store(keys, object_sizes)
        except Exception as error:
            state.failure_codes.update((key, None) for key in keys)
            state.errors_by_key.update((key, error) for key in keys)
            self._revoke_store_keys(state, keys, {})
            return
        opened_keys = []
        for key, code in zip(keys, result_codes, strict=True):
            if code == 0:
                opened_keys.append(key)
            else:
                state.failure_codes[key] = code
                state.errors_by_key[key] = RuntimeError(
                    f"Store session start failed for key {key!r}; result code {code}"
                )
        state.opened_keys = tuple(opened_keys)
        if not state.opened_keys:
            return
        selected = batch if len(state.opened_keys) == len(keys) else batch.select_keys(set(state.opened_keys))
        try:
            state.keys_by_layer = self._prepare_store_layers(selected)
        except Exception as error:
            state.failure_codes.update((key, None) for key in state.opened_keys)
            state.errors_by_key.update((key, error) for key in state.opened_keys)

    def _store_layer(self, source_ready: Any, layer_id: int, state: _StoreState) -> None:
        keys = state.keys_by_layer.get(layer_id)
        if keys:
            result = self._copy_store_layer(source_ready, layer_id)
            self._record_store_result(state, keys, result)

    def _finish_store(
        self, opening: Future[None], layers: dict[int, Future[None]], state: _StoreState
    ) -> tuple[StoreCompletion, ...]:
        # Session supplies terminal Futures; this method interprets their business outcomes.
        try:
            opening.result()
        except (BrokenExecutor, CancelledError):
            raise
        except Exception as error:
            # Layer jobs depend on opening, so no source was handed off after its failure.
            self._store_plan = None
            if state.opened_keys:
                try:
                    self._backend_sessions.revoke_store(list(state.opened_keys))
                except BaseException:
                    logger.exception("Failed to revoke Layerwise Store sessions after preparation failed")
            return _store_command_completions(state.commands, error)
        for layer_id, future in layers.items():
            try:
                future.result()
            except (BrokenExecutor, CancelledError):
                raise
            except Exception as error:
                self._record_store_result(state, state.keys_by_layer[layer_id], LayerStoreResult(None, False, error))
        self._store_plan = None
        if state.batch is None:
            return _store_command_completions(state.commands)

        # Preparation succeeded and every submitted Future was observed; no second layer ledger is needed.
        missing_layers = state.keys_by_layer.keys() - layers.keys()
        if missing_layers:
            incomplete = RuntimeError(f"Layerwise Store did not reach physical layers {sorted(missing_layers)}")
            for key in state.opened_keys:
                state.failure_codes.setdefault(key, None)
                state.errors_by_key.setdefault(key, incomplete)
        commit_keys = [key for key in state.opened_keys if key not in state.failure_codes]
        final_codes: dict[str, int | None] = {}
        if commit_keys:
            try:
                result_codes = self._backend_sessions.commit_store(commit_keys)
            except Exception as error:
                state.failure_codes.update((key, None) for key in commit_keys)
                state.errors_by_key.update((key, error) for key in commit_keys)
            else:
                for key, code in zip(commit_keys, result_codes, strict=True):
                    final_codes[key] = code
                    if code != 0:
                        state.failure_codes[key] = code
        revoke_keys = [key for key in state.opened_keys if key in state.failure_codes]
        self._revoke_store_keys(state, revoke_keys, final_codes)
        observed_layers = state.keys_by_layer.keys() & layers.keys()
        return self._build_store_completions(state, observed_layers, final_codes)

    @staticmethod
    def _record_store_result(state: _StoreState, keys: list[str], result: LayerStoreResult) -> None:
        # The concrete Worker normalizes result axes. Keep a scalar all-success result constant-time.
        if isinstance(result.result_codes, tuple):
            codes = iter(result.result_codes)
            releases = iter(cast(tuple[bool, ...], result.source_release_confirmed))
        else:
            if result.result_codes == 0 and result.source_release_confirmed and result.error is None:
                return
            codes = repeat(result.result_codes, len(keys))
            releases = repeat(cast(bool, result.source_release_confirmed), len(keys))
        for key, code, released in zip(keys, codes, releases, strict=True):
            if code != 0:
                state.failure_codes.setdefault(key, code)
            if not released:
                state.unsafe_keys.add(key)
            if result.error is not None:
                state.errors_by_key.setdefault(key, result.error)

    def _revoke_store_keys(self, state: _StoreState, keys: list[str], final_codes: dict[str, int | None]) -> None:
        if not keys:
            return
        try:
            result_codes = self._backend_sessions.revoke_store(keys)
        except Exception as error:
            for key in keys:
                state.errors_by_key.setdefault(key, error)
        else:
            for key, code in zip(keys, result_codes, strict=True):
                if code != 0:
                    # A failed revoke must not replace a commit result already reported by Backend.
                    final_codes.setdefault(key, code)
                    state.errors_by_key.setdefault(
                        key, RuntimeError(f"Store session revocation was not confirmed; result code {code}")
                    )

    @staticmethod
    def _build_store_completions(
        state: _StoreState, observed_layers: set[int], final_codes: dict[str, int | None]
    ) -> tuple[StoreCompletion, ...]:
        batch = state.batch
        assert batch is not None
        opened_keys = set(state.opened_keys)
        evidence_by_request: list[list[TransferEvidence]] = [[] for _ in state.commands]
        failed_requests = set()
        errors_by_request: dict[int, Exception] = {}
        # Derive ownership and final source evidence together, once, instead of caching both per request.
        for group in batch.groups:
            physical_layer_ids = tuple(layer_id for layer_id in group.physical_layer_ids if layer_id in observed_layers)
            for request_index in range(len(state.commands)):
                start = int(group.request_splits[request_index])
                end = int(group.request_splits[request_index + 1])
                for axis_index, axis in enumerate(group.key_axes):
                    axis_offset = axis_index * group.row_count
                    for row_index in range(start, end):
                        object_index = axis_offset + row_index
                        if group.selected_objects is not None and not group.selected_objects[object_index]:
                            continue
                        key = axis[row_index]
                        if key in state.failure_codes:
                            failed_requests.add(request_index)
                        if key in state.errors_by_key:
                            errors_by_request.setdefault(request_index, state.errors_by_key[key])
                        if key not in opened_keys or not physical_layer_ids:
                            continue
                        source = TransferSource(group, object_index, row_index, request_index, physical_layer_ids, key)
                        evidence_by_request[request_index].append(
                            TransferEvidence(
                                source, final_codes.get(key, state.failure_codes.get(key)), key not in state.unsafe_keys
                            )
                        )
        return tuple(
            StoreCompletion(
                command.request_id,
                StoreEvidence(
                    tuple(evidence),
                    index not in failed_requests and index not in errors_by_request,
                    all(item.source_release_confirmed is True for item in evidence),
                    errors_by_request.get(index),
                ),
                command.store_job_id,
            )
            for index, (command, evidence) in enumerate(zip(state.commands, evidence_by_request, strict=True))
        )

    def _submit_step_store(self, context: KVPoolStepContext, *, fence: bool) -> None:
        self._finalize_layerwise_store(context)
        if fence:
            self.fence_previous_store()

    def _finalize_layerwise_store(self, context: KVPoolStepContext) -> None:
        if self._pending_store_batch is not None:
            raise RuntimeError("Previous Store invocation has not reached its fence")
        batch = self._prepare_store_close()
        assert batch is not None
        self._track_store_batch(batch)
        context.store_submitted = True

    def _prepare_store_close(self) -> PendingStore | None:
        state = self._store_state
        if state is None:
            return None
        assert self._store_opening is not None
        future = self._get_session().finish_store(self._store_opening, self._finish_store, state)
        # Seal submission. Queued tasks retain this invocation's state, even after end_step().
        self._store_state = None
        self._store_opening = None
        return PendingStore(
            future,
            frozenset(command.request_id for command in state.commands),
            frozenset(command.store_job_id for command in state.commands if command.store_job_id is not None),
        )

    def _submit_store_layer(self, layer_name: str) -> None:
        layer_id = self._layer_ids_by_name.get(layer_name)
        if layer_id is None or not self._can_store_layer(layer_id):
            return
        assert self._store_state is not None and self._store_opening is not None
        self._get_session().store_layer(
            layer_id,
            self._store_opening,
            self._record_source_ready,
            self._store_layer,
            self._store_state,
        )

    def _can_store_layer(self, layer_id: int) -> bool:
        state, opening = self._store_state, self._store_opening
        if state is None or opening is None:
            return False
        if not opening.done():
            # The source boundary cannot be recorded later on the background thread.
            return True
        if opening.cancelled() or opening.exception() is not None:
            return False  # Finalization owns preparation errors; no source needs handing off.
        return layer_id in state.keys_by_layer

    def _wait_for_layer_load(self, layer_name: str) -> tuple[LoadCompletion, ...]:
        context = self._active_step
        if context is None:
            raise RuntimeError("KV Pool step has not begun")
        layer_id = self._layer_ids_by_name.get(layer_name)
        if layer_id is None:
            return ()
        store_at_attention_entry = layer_id in self._attention_store_layers and self._can_store_layer(layer_id)
        if self._session is None and not store_at_attention_entry:
            return ()
        try:
            session = self._get_session()
            result = session.wait_for_layer(layer_id, self._copy_load_layer, self._finish_load)
            if store_at_attention_entry:
                session.bind_store_to_compute_start(layer_id, self.save_layer, layer_name)
            return result or ()
        except (BrokenExecutor, CancelledError):
            raise
        except Exception:
            if self._load_batch is None:
                raise
            logger.exception("Layerwise Load failed at physical layer %d", layer_id)
            self._abort_load()
            return _failed_load_completions(self._load_batch, layer_id)

    def end_step(self) -> None:
        if self._session is not None or self._backend_sessions.has_load_sessions:
            finish_load = self._finish_load if self._backend_sessions.has_load_sessions else None
            self._get_session().end_step(finish_load)
        closing_batch = self._prepare_store_close()
        if closing_batch is not None:
            self._track_store_batch(closing_batch)
        self._session = None
        self._store_state = None
        self._load_batch = None
        super().end_step()

    def close(self) -> None:
        if self._session is not None or self._backend_sessions.has_load_sessions:
            finish_load = self._finish_load if self._backend_sessions.has_load_sessions else None
            self._get_session().end_step(finish_load, wait=False)
        super().close()

    def _check_execution_results(self) -> None:
        if self._session is not None:
            self._session.wait_load_finished()


def collect_object_sizes(batch: KVTransferBatch) -> dict[str, int]:
    object_sizes: dict[str, int] = {}
    for group in batch.groups:
        for key in group.selected_keys():
            previous = object_sizes.setdefault(key, group.object_size)
            if previous != group.object_size:
                raise ValueError(f"Layerwise key {key!r} has inconsistent object sizes")
    return object_sizes


def compile_layer_ids_by_name(topology: KVPoolTopology) -> dict[str, int]:
    layer_ids_by_name: dict[str, int] = {}
    for group in topology.groups:
        for layer in group.layers:
            for layer_name in layer.layer_names:
                previous = layer_ids_by_name.setdefault(layer_name, layer.physical_layer_id)
                if previous != layer.physical_layer_id:
                    raise ValueError(
                        f"Layer {layer_name!r} maps to both physical layers {previous} and {layer.physical_layer_id}"
                    )
    return layer_ids_by_name


def _load_session_failures(batch: KVTransferBatch, failed_codes: dict[str, int]) -> tuple[LoadCompletion, ...]:
    evidence_by_request: list[list[TransferEvidence]] = [[] for _ in batch.request_ids]
    for group in batch.groups:
        selected = (
            range(group.object_count)
            if group.selected_objects is None
            else np.flatnonzero(group.selected_objects).tolist()
        )
        for object_index in selected:
            axis_index, row_index = divmod(object_index, group.row_count)
            code = failed_codes.get(group.key_axes[axis_index][row_index], 0)
            if code != 0:
                source = group.source(object_index, None)
                evidence_by_request[source.request_index].append(TransferEvidence(source, code))
    return tuple(
        LoadCompletion(request_id, tuple(evidence))
        for request_id, evidence in zip(batch.request_ids, evidence_by_request, strict=True)
        if evidence
    )


def _attention_store_layer_ids(topology: KVPoolTopology) -> frozenset[int]:
    """Match production's KeyRange Hybrid boundary, excluding recurrent physical layers."""

    if len(topology.groups) <= 1:
        return frozenset()
    recurrent_layers = set()
    for group in topology.groups:
        for layer in group.layers:
            specs = (
                tuple(group.kv_cache_spec.kv_cache_specs[name] for name in layer.layer_names)
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else (group.kv_cache_spec,)
            )
            if any(isinstance(spec, MambaSpec) for spec in specs):
                recurrent_layers.add(layer.physical_layer_id)
    return frozenset(
        layer.physical_layer_id
        for group in topology.transfer_groups
        for layer in group.layers
        if layer.physical_layer_id not in recurrent_layers
    )


def _is_store_leader(topology: KVPoolTopology) -> bool:
    """Resolve the rank that owns each Layerwise object before request traffic."""

    if topology.dcp_size <= 1:
        return topology.tp_rank % topology.put_step == 0
    dcp_rank = topology.transfer_groups[0].key_metadata.dcp_rank
    head = topology.tp_rank // topology.put_step
    peers = tuple(
        rank
        for rank in range(head * topology.put_step, (head + 1) * topology.put_step)
        if rank % topology.dcp_size == dcp_rank
    )
    return not peers or topology.tp_rank == min(peers)


__all__ = ("LayerwiseWorker",)
