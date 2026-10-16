"""Shared Worker route for layer-restricted Backend sessions."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, CancelledError, Future
from dataclasses import dataclass, field
from itertools import repeat
from typing import Any, TypeAlias, cast

import numpy as np
from vllm.logger import logger
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from ....attention_fence import reset_attention_compute_start_gate
from ...execution.executor import Executor
from ...execution.session import Session
from ...projection import GVALayerwiseProjection, LayerwiseProjection, LayerwiseProjectionBinder
from ...projection.layerwise.gva import gva_local_keys, gva_lookup_keys
from ...projection.layerwise.key_range import key_range_local_keys, key_range_lookup_keys
from ...projection.store import StoreCandidateRows, store_candidate_rows
from ...protocol.lookup import LookupRequest, LookupResult
from ...protocol.transfer import CheckpointStoreCommand, LoadCommand, StoreCommand
from ...topology import KVPoolTopology
from ..base import KVPoolWorker, _store_command_completions
from ..io import GVABackendIO, KeyRangeBackendIO
from ..io.io import _failed_load_completions
from ..resources import KVPoolResources
from ..transfer.batch import KeyAxes, KVGroupBatch, KVTransferBatch, TransferSource
from ..transfer.evidence import LayerStoreResult, LoadCompletion, StoreCompletion, StoreEvidence, TransferEvidence
from ..transfer.rows import (
    BlockRows,
    load_block_rows,
    lookup_chunk_rows,
)
from ..transfer.state import (
    KVPoolStepContext,
    PendingStore,
    StoreCandidates,
    StoreGroupCandidates,
    concat_uint64,
    empty_candidate_rows,
    empty_rows,
    merge_key_axes,
    readonly_indices,
    select_store_candidate_objects,
    store_candidate_keys,
)

_LayerwiseBackendIO: TypeAlias = GVABackendIO | KeyRangeBackendIO


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
    """Own the shared Layerwise request, session, and per-layer lifecycle."""

    def __init__(
        self,
        topology: KVPoolTopology,
        projection_binder: LayerwiseProjectionBinder,
        resources: KVPoolResources,
        backend_io: _LayerwiseBackendIO,
        *,
        store_enabled: bool = True,
        layerwise_prefetch_layers: int = 2,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._layerwise_projection_binder = projection_binder
        self._layerwise_projection: LayerwiseProjection | None = None
        self._layerwise_backend_io = backend_io
        self._layerwise_store_leader = _is_store_leader(topology)
        self._start_gate_factory = start_gate_factory
        self._prefetch_layers = layerwise_prefetch_layers
        self._layer_ids_by_name = compile_layer_ids_by_name(topology)
        self._session: Session[tuple[LoadCompletion, ...]] | None = None
        self._store_state: _StoreState | None = None
        self._store_opening: Future[None] | None = None
        self._load_batch: KVTransferBatch | None = None
        self._load_session_keys: tuple[str, ...] = ()
        self._load_initial_completions: tuple[LoadCompletion, ...] = ()
        self._attention_store_layers = (
            _attention_store_layer_ids(topology)
            if isinstance(backend_io, KeyRangeBackendIO) and store_enabled
            else frozenset()
        )
        self._load_executor = Executor("KVPoolLayerwiseLoadExecutor", backend_io.initialize_thread)
        self._store_executor = (
            Executor("KVPoolLayerwiseStoreExecutor", backend_io.initialize_thread) if store_enabled else None
        )
        super().__init__(
            topology,
            resources,
            backend_io,
            tuple(executor for executor in (self._store_executor, self._load_executor) if executor is not None),
            store_enabled=store_enabled,
            source_ready_event_factory=source_ready_event_factory,
        )

    def _bind_projection(self, registration: dict[str, Any]) -> None:
        projection = self._layerwise_projection_binder.bind(**registration)
        self._bind_backend_projection(projection)
        self._layerwise_projection = projection

    def _submit_load(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        object_sizes = collect_object_sizes(batch)
        if not object_sizes:
            return ()
        self._load_batch = batch
        self._load_session_keys = tuple(object_sizes)
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
        result_codes = self._layerwise_backend_io.start_load_sessions(list(object_sizes), list(object_sizes.values()))
        codes_by_key = dict(zip(object_sizes, result_codes, strict=True))
        self._load_session_keys = tuple(key for key, code in codes_by_key.items() if code == 0)
        self._load_initial_completions = _load_session_failures(batch, codes_by_key)
        if not self._load_session_keys:
            return ()
        selected = batch.select_keys(set(self._load_session_keys))
        keys_by_layer = self._layerwise_backend_io.prepare_load_layers(selected)
        return tuple(sorted(layer_id for layer_id, keys in keys_by_layer.items() if keys))

    def _finish_load(self) -> None:
        if self._load_session_keys:
            self._layerwise_backend_io.finish_load_sessions(list(self._load_session_keys))
            self._load_session_keys = ()

    def _collect_load_completions(self) -> tuple[LoadCompletion, ...]:
        if self._session is not None and (pending := self._session.pending_load_layers):
            self._abort_load()
            raise RuntimeError(f"Layerwise Load did not reach physical layers {list(pending)}")
        return ()

    def _abort_load(self, *, wait: bool = True) -> None:
        if self._session is not None or self._load_session_keys:
            finish = self._finish_load if self._load_session_keys else None
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

    @property
    def _bound_layerwise_projection(self) -> LayerwiseProjection:
        if self._layerwise_projection is None:
            raise RuntimeError("Layerwise projection is unavailable before cache registration")
        return self._layerwise_projection

    @abstractmethod
    def _bind_backend_projection(self, projection: LayerwiseProjection) -> None:
        """Bind the concrete projection to the selected Backend I/O adapter."""

    def lookup(self, request: LookupRequest) -> LookupResult:
        projection = self._bound_layerwise_projection
        if request.transfer_group_ids != projection.group_ids:
            raise ValueError(
                f"Lookup groups {request.transfer_group_ids} do not match configured groups {projection.group_ids}"
            )
        masks = projection.reachability.select_for_lookup(request.block_hashes, request.query_range)
        observations = []
        for group_id, mask in zip(projection.group_ids, masks, strict=True):
            group = projection.groups[group_id]
            starts, counts, hashes = lookup_chunk_rows(
                request.query_range.end_token,
                request.block_hashes,
                block_size=group.block_size,
                hash_block_size=group.hash_block_size,
                start_token=request.query_range.start_token,
                mask=mask,
            )
            keys = self._lookup_keys(projection, group_id, hashes)
            try:
                lookup_keys = [key for axis in keys for key in axis]
                observed = self._backend_io.exists(lookup_keys)
            except Exception as error:
                logger.error("Remote Lookup failed. type=%s, error=%s", type(error).__name__, error)
                return LookupResult(0)
            row_count = len(hashes)
            available = tuple(
                all(observed[axis_index * row_count + row_index] for axis_index in range(len(keys)))
                for row_index in range(row_count)
            )
            observations.append(((starts, counts, hashes), available))
        reachable = projection.reachability.resolve_available_end(
            request.query_range,
            request.block_hashes,
            observations,
        )
        return LookupResult(reachable.end_token, reachable.tail_key_boundaries)

    @staticmethod
    def _lookup_keys(projection: LayerwiseProjection, group_id: int, hashes) -> KeyAxes:
        if isinstance(projection, GVALayerwiseProjection):
            return gva_lookup_keys(projection.groups[group_id], hashes)
        return key_range_lookup_keys(projection.groups[group_id], hashes)

    @staticmethod
    def _local_keys(projection: LayerwiseProjection, group_id: int, hashes) -> KeyAxes:
        if isinstance(projection, GVALayerwiseProjection):
            return gva_local_keys(projection.groups[group_id], hashes)
        return key_range_local_keys(projection.groups[group_id], hashes)

    def _build_load_batch(self, commands: tuple[LoadCommand, ...]) -> KVTransferBatch:
        projection = self._bound_layerwise_projection
        masks_by_request = tuple(
            projection.reachability.select_for_load(command.block_hashes, command.load_range) for command in commands
        )
        rows_by_request: list[dict[int, BlockRows]] = []
        for command, masks in zip(commands, masks_by_request, strict=True):
            boundaries = {item.group_id: item.boundary_token for item in command.tail_key_boundaries}
            if len(boundaries) != len(command.tail_key_boundaries):
                raise ValueError("Load contains duplicate tail-key boundaries for one cache group")
            unknown = set(boundaries).difference(projection.group_ids)
            if unknown:
                raise ValueError(f"Load tail-key boundaries contain unknown cache groups {sorted(unknown)}")
            rows_by_request.append(
                {
                    group_id: load_block_rows(
                        command.load_range.end_token,
                        command.block_hashes,
                        command.block_ids_by_group[group_id],
                        block_size=projection.groups[group_id].block_size,
                        hash_block_size=projection.groups[group_id].hash_block_size,
                        start_token=command.load_range.start_token,
                        mask=mask,
                        tail_boundary_token=boundaries.get(group_id),
                    )
                    for group_id, mask in zip(projection.group_ids, masks, strict=True)
                }
            )
        return self._assemble_load_batch(tuple(command.request_id for command in commands), rows_by_request)

    def _build_store_candidates(self, commands: tuple[StoreCommand, ...]) -> StoreCandidates:
        projection = self._bound_layerwise_projection
        rows_by_group: list[list[StoreCandidateRows]] = [[] for _group_id in projection.group_ids]
        keys_by_group: list[list[KeyAxes]] = [[] for _group_id in projection.group_ids]
        for command in commands:
            for group_index, (group_id, rows) in enumerate(
                zip(projection.group_ids, self._store_candidate_rows(command), strict=True)
            ):
                rows_by_group[group_index].append(rows)
                keys_by_group[group_index].append(self._local_keys(projection, group_id, rows[1]))
        return StoreCandidates(
            tuple(
                StoreGroupCandidates(group_id, tuple(rows), merge_key_axes(group_id, keys_by_group[group_index]))
                for group_index, (group_id, rows) in enumerate(zip(projection.group_ids, rows_by_group, strict=True))
            )
        )

    def _store_candidate_rows(self, command: StoreCommand) -> tuple[StoreCandidateRows, ...]:
        projection = self._bound_layerwise_projection
        if isinstance(command, CheckpointStoreCommand):
            raise ValueError("Layerwise transfer does not support state checkpoint Store")

        if not self._layerwise_store_leader:
            return tuple(empty_candidate_rows() for _group_id in projection.group_ids)
        masks = projection.reachability.select_for_store(
            command.block_hashes,
            command.store_range,
            command.num_prompt_tokens,
        )
        return tuple(
            store_candidate_rows(
                command.store_range.end_token,
                command.block_hashes,
                command.block_ids_by_group[group_id],
                block_size=projection.groups[group_id].block_size,
                hash_block_size=projection.groups[group_id].hash_block_size,
                start_token=command.store_range.start_token,
                mask=mask,
            )
            for group_id, mask in zip(projection.group_ids, masks, strict=True)
        )

    def _assemble_load_batch(
        self,
        request_ids: tuple[str, ...],
        rows_by_request: list[dict[int, BlockRows]],
    ) -> KVTransferBatch:
        projection = self._bound_layerwise_projection
        groups = []
        for group_id in projection.group_ids:
            block_parts = []
            count_parts = []
            request_splits = [0]
            for rows_by_group in rows_by_request:
                rows = rows_by_group.get(group_id, empty_rows())
                counts, _, block_ids = rows
                block_parts.append(block_ids)
                count_parts.append(counts)
                request_splits.append(request_splits[-1] + len(block_ids))
            key_axes = merge_key_axes(
                group_id,
                [
                    self._local_keys(projection, group_id, rows_by_group.get(group_id, empty_rows())[1])
                    for rows_by_group in rows_by_request
                ],
            )
            topology = self._groups[group_id]
            groups.append(
                KVGroupBatch(
                    group_id,
                    concat_uint64(block_parts),
                    concat_uint64(count_parts),
                    key_axes,
                    readonly_indices(request_splits),
                    tuple(layer.physical_layer_id for layer in topology.layers),
                    projection.groups[group_id].object_size,
                )
            )
        return KVTransferBatch(request_ids, tuple(groups))

    def _store_group_layout(self, group_id: int) -> tuple[int, tuple[tuple[int, ...], ...] | None]:
        return self._bound_layerwise_projection.groups[group_id].object_size, None

    @property
    def _transfer_group_count(self) -> int:
        return len(self._bound_layerwise_projection.group_ids)

    def _prepare_step_store(self, commands: tuple[StoreCommand, ...]) -> None:
        # Backend IO has one Store plan; a new Store waits, an empty decode does not.
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
        keys = list(batch.selected_keys())
        object_sizes = [group.object_size for group in batch.groups for _ in group.selected_keys()]
        io = self._layerwise_backend_io
        try:
            result_codes = io.start_store_sessions(keys, object_sizes)
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
            state.keys_by_layer = io.prepare_store_layers(selected)
        except Exception as error:
            state.failure_codes.update((key, None) for key in state.opened_keys)
            state.errors_by_key.update((key, error) for key in state.opened_keys)

    def _store_layer(self, source_ready: Any, layer_id: int, state: _StoreState) -> None:
        keys = state.keys_by_layer.get(layer_id)
        if keys:
            result = self._layerwise_backend_io.store_layer(source_ready, layer_id)
            self._record_store_result(state, keys, result)

    def _finish_store(
        self, opening: Future[None], layers: dict[int, Future[None]], state: _StoreState
    ) -> tuple[StoreCompletion, ...]:
        # Session supplies terminal Futures; this method interprets their business outcomes.
        io = self._layerwise_backend_io
        try:
            opening.result()
        except (BrokenExecutor, CancelledError):
            raise
        except Exception as error:
            # Layer jobs depend on opening, so no source was handed off after its failure.
            if state.opened_keys:
                try:
                    io.revoke_store_sessions(list(state.opened_keys))
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
                result_codes = io.commit_store_sessions(commit_keys)
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
        # IO normalizes result axes. Keep a scalar all-success result constant-time.
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
            result_codes = self._layerwise_backend_io.revoke_store_sessions(keys)
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
            result = session.wait_for_layer(layer_id, self._layerwise_backend_io.load_layer, self._finish_load)
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
        if self._session is not None:
            self._session.clear_callbacks()
        self._abort_load()
        closing_batch = self._prepare_store_close()
        if closing_batch is not None:
            self._track_store_batch(closing_batch)
        self._session = None
        self._store_state = None
        self._load_batch = None
        super().end_step()

    def close(self) -> None:
        if self._session is not None:
            self._session.clear_callbacks()
        self._abort_load(wait=False)
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


def _load_session_failures(batch: KVTransferBatch, codes_by_key: dict[str, int]) -> tuple[LoadCompletion, ...]:
    evidence_by_request: list[list[TransferEvidence]] = [[] for _ in batch.request_ids]
    for group in batch.groups:
        selected = (
            range(group.object_count)
            if group.selected_objects is None
            else np.flatnonzero(group.selected_objects).tolist()
        )
        for object_index in selected:
            source = group.source(object_index, None)
            code = codes_by_key[source.key]
            if code != 0:
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
