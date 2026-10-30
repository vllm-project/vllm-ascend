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
from ...backend.naming import WireKeyBuilder
from ...infrastructure.coordinator import Coordinator
from ...infrastructure.execution import Executor, Session
from ...infrastructure.mapping import (
    LayerTable,
    LocalTable,
    RemoteTable,
    build_lookup_keys,
    build_mappings,
    map_keys,
    select_layers,
)
from ...infrastructure.resources import Resources
from ...infrastructure.topology import KVPoolTopology
from ...protocol.messages import CheckpointStoreCommand, KVTransferStep, StoreCommand
from ..base import KVPoolStepContext, KVPoolWorker, PendingStore
from ..transfer.batch import (
    KeysByPartition,
    KVTransferBatch,
    StoreGroupTransferCandidates,
    StoreTransferCandidates,
    TransferSource,
    select_store_transfer_candidates,
    store_transfer_candidate_keys,
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
    StoreTransferRecords,
    merge_keys_by_partition,
    store_transfer_records,
)
from .transfer import LayerStoreResult, LayerTransferPlan, PreparedLayerLoad, bind_layer_memberships


@dataclass(slots=True)
class _LoadState:
    """Retain this Load's ownership and preparation through result claiming.

    Remote session cleanup can precede the model's final result claim. Keep
    the prepared copy inputs until end_step() detaches this operation.
    """

    batch: KVTransferBatch
    prepared: PreparedLayerLoad | None = None
    initial_completions: tuple[LoadCompletion, ...] = ()


@dataclass(slots=True)
class _StoreState:
    """One Store's inputs and accumulated outcomes, retained by its queued tasks.

    The serial Store executor mutates preparation and outcomes; the caller
    retains pending during handoff. failure_codes contains failures only;
    an absent key is not yet a successful publication.
    """

    commands: tuple[StoreCommand, ...]
    pending: PendingStore | None = None
    batch: KVTransferBatch | None = None
    plan: LayerTransferPlan | None = None
    opened_keys: tuple[str, ...] = ()
    selected_keys_by_layer: dict[int, list[str]] = field(default_factory=dict)
    failure_codes: dict[str, int | None] = field(default_factory=dict)
    unsafe_keys: set[str] = field(default_factory=set)
    errors_by_key: dict[str, Exception] = field(default_factory=dict)


class LayerwiseWorker(KVPoolWorker):
    """Own request and layer scheduling; Backend owns the remote object lifecycle."""

    def __init__(
        self,
        topology: KVPoolTopology,
        resources: Resources,
        backend_sessions: ObjectSessions,
        *,
        coordinator: Coordinator,
        kv_cache_layout: str = "NHD",
        store_enabled: bool = True,
        layerwise_prefetch_layers: int = 2,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
        key_builder: WireKeyBuilder | None = None,
    ) -> None:
        self._key_builder = key_builder
        self._backend_sessions = backend_sessions
        resources.bind_backend_sessions(backend_sessions)
        self._store_writers = tuple(min(peers) for peers in topology.context_replica_groups)
        self._layerwise_store_leader = topology.global_rank in self._store_writers
        self._start_gate_factory = start_gate_factory
        self._prefetch_layers = layerwise_prefetch_layers
        self._layer_ids_by_name = compile_layer_ids_by_name(topology)
        self._session: Session[tuple[LoadCompletion, ...]] | None = None
        self._store_state: _StoreState | None = None
        self._store_opening: Future[None] | None = None
        self._load_state: _LoadState | None = None
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
            coordinator=coordinator,
            kv_cache_layout=kv_cache_layout,
            source_ready_event_factory=source_ready_event_factory,
        )

    def begin_step(self, step: KVTransferStep) -> None:
        context = self._new_step_context(step)
        if self._store_enabled and step.store.commands:
            # The common fence retains its own failures. An empty Decode Step does not wait.
            if self._pending_store_batches:
                self.fence_previous_store()
            try:
                state = _StoreState(step.store.commands)
                self._store_state = state
                assert self._store_executor is not None
                self._store_opening = self._store_executor.submit(self._open_store, state)
                # Preserve the all-ready handoff without making it a per-layer readiness rule.
                if step.store.all_sources_ready:
                    self._seal_store(context)
            except Exception as error:
                self._store_fatal_error = error
                raise
        self._active_step = context

    @KVPoolWorker._with_active_step
    def finish_step(self, context: KVPoolStepContext) -> None:
        self._raise_store_fatal_error()
        if not self._store_enabled or not context.step.store.commands or context.store_submitted:
            return
        try:
            self._seal_store(context)
        except Exception as error:
            self._store_fatal_error = error
            raise
        self.fence_previous_store()

    @KVPoolWorker._with_active_step
    def wait_for_layer_load(self, context: KVPoolStepContext, layer_name: str) -> None:
        for completion in self._wait_for_layer_load(layer_name):
            self._record_load_completion(context, completion)
        self._raise_load_error(context)

    @KVPoolWorker._with_active_step
    def save_layer(self, _context: KVPoolStepContext, layer_name: str) -> None:
        self._raise_store_fatal_error()
        try:
            self._submit_store_layer(layer_name)
        except Exception as error:
            self._store_fatal_error = error
            raise

    def _bind_mappings(self) -> None:
        kv_caches = self._resources.kv_caches
        if kv_caches is None:
            raise RuntimeError("KV tensors must be retained before binding Layerwise mappings")
        self._local: dict[int, LocalTable] = {}
        self._layer_slices: dict[int, LayerTable] = {}
        self._remote: dict[int, RemoteTable] = {}
        self._physical_layer_ids_by_partition = {}
        for group_id, group in self._groups.items():
            keys, local, layer_slices = build_mappings(
                self._topology,
                kv_caches,
                group_id=group_id,
                kv_cache_layout=self._kv_cache_layout,
                key_builder=self._key_builder,
                layerwise=True,
            )
            if not any(len(segments["bases"]) for segments in local.values()):
                raise ValueError("KV cache group has no registered memory entries")
            assert layer_slices is not None
            self._keys[group_id] = keys
            self._local[group_id] = local
            self._layer_slices[group_id] = layer_slices
            head_count = (
                self._topology.config.tp_size
                if group.uses_align_state
                else self._topology.config.tp_partition.key_rank_count
            )
            self._lookup_key_parts[group_id] = (
                build_lookup_keys(self._topology, group_id)
                if self._key_builder is None
                else self._key_builder.lookup_parts(group_id, head_count)
            )
            layers_by_pp = select_layers(group, self._topology.config.pp_partition.target_layer_counts)
            self._physical_layer_ids_by_partition[group_id] = tuple(
                tuple(layer.physical_layer_id for layer in layers_by_pp[pp]) for pp, _, _ in keys
            )
        self._layer_memberships = bind_layer_memberships(self._local, self._layer_slices)

    def _group_object_layout(self, group_id: int) -> tuple[int, tuple[tuple[int, ...], ...]]:
        # Admission excludes TP mismatch and target-PP splitting: one local object per group.
        remote = next(iter(self._remote[group_id].values()))
        return remote["size_bytes"], self._physical_layer_ids_by_partition[group_id]

    @abstractmethod
    def _prepare_load_layers(self, batch: KVTransferBatch) -> PreparedLayerLoad: ...

    @abstractmethod
    def _prepare_store_layers(self, batch: KVTransferBatch) -> LayerTransferPlan: ...

    @abstractmethod
    def _copy_store_layer(
        self, source_ready_event: Any, layer_id: int, plan: LayerTransferPlan
    ) -> LayerStoreResult: ...

    def _submit_load(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        object_sizes = collect_object_sizes(batch)
        if not object_sizes:
            return ()
        state = _LoadState(batch)
        self._load_state = state
        try:
            self._load_executor.submit(self._open_load, state, object_sizes).result()
            order = (
                tuple(sorted(layer_id for layer_id, keys in state.prepared.plan.selected_keys_by_layer.items() if keys))
                if state.prepared is not None
                else ()
            )
            if order:
                self._get_session().load_order = order
        except (BrokenExecutor, CancelledError):
            raise
        except Exception:
            logger.exception("Layerwise Load preparation failed for requests %s", batch.request_ids)
            self._abort_load()
            return _failed_load_completions(batch, None)
        return state.initial_completions

    def _open_load(self, state: _LoadState, object_sizes: dict[str, int]) -> None:
        batch = state.batch
        result_codes = self._backend_sessions.start_load(list(object_sizes), list(object_sizes.values()))
        opened_keys = set()
        failed_codes = {}
        for key, code in zip(object_sizes, result_codes, strict=True):
            if code == 0:
                opened_keys.add(key)
            else:
                failed_codes[key] = code
        state.initial_completions = _load_session_failures(batch, failed_codes) if failed_codes else ()
        if not opened_keys:
            return
        selected = batch if not failed_codes else batch.select_keys(opened_keys)
        state.prepared = self._prepare_load_layers(selected)

    def _collect_load_completions(self) -> tuple[LoadCompletion, ...]:
        if self._session is not None and (pending := self._session.pending_load_layers):
            self._abort_load()
            raise RuntimeError(f"Layerwise Load did not reach physical layers {list(pending)}")
        return ()

    def _abort_load(self, *, wait: bool = True) -> None:
        if self._session is not None or self._backend_sessions.has_load_sessions:
            finish = self._backend_sessions.finish_load if self._backend_sessions.has_load_sessions else None
            self._get_session().abort_load(finish, wait=wait)

    def _get_session(self) -> Session[tuple[LoadCompletion, ...]]:
        if self._session is None:
            store_submit = self._store_executor.submit if self._store_executor is not None else None
            self._session = Session(
                self._load_executor.submit, store_submit, self._prefetch_layers, self._start_gate_factory
            )
        return self._session

    def _build_store_candidates(self, commands: tuple[StoreCommand, ...]) -> StoreTransferCandidates:
        if any(isinstance(command, CheckpointStoreCommand) for command in commands):
            raise ValueError("Layerwise transfer does not support state checkpoint Store")
        if not self._layerwise_store_leader:
            return StoreTransferCandidates(())
        transfer_records_by_group: list[list[StoreTransferRecords]] = [
            [] for _group_id in self._topology.config.transfer_group_ids
        ]
        keys_by_group: list[list[KeysByPartition]] = [[] for _group_id in self._topology.config.transfer_group_ids]
        for command in commands:
            for group_index, (group_id, rows) in enumerate(
                zip(self._topology.config.transfer_group_ids, self._store_transfer_records(command), strict=True)
            ):
                transfer_records_by_group[group_index].append(rows)
                keys_by_group[group_index].append(
                    tuple(map_keys(parts, rows[1]) for parts in self._keys[group_id].values())
                )
        return StoreTransferCandidates(
            tuple(
                StoreGroupTransferCandidates(
                    group_id, tuple(rows), merge_keys_by_partition(group_id, keys_by_group[group_index])
                )
                for group_index, (group_id, rows) in enumerate(
                    zip(self._topology.config.transfer_group_ids, transfer_records_by_group, strict=True)
                )
            )
        )

    def _store_transfer_records(self, command: StoreCommand) -> tuple[StoreTransferRecords, ...]:
        assert not isinstance(command, CheckpointStoreCommand)
        masks = self._coordinator.select_for_store(command.start_token, command.end_token, command.num_prompt_tokens)
        return tuple(
            store_transfer_records(
                command.end_token,
                command.block_hashes,
                command.block_ids_by_group[group_id],
                block_size=self._groups[group_id].block_size,
                hash_block_size=self._topology.config.hash_block_size,
                start_token=command.start_token,
                mask=mask,
            )
            for group_id, mask in zip(self._topology.config.transfer_group_ids, masks, strict=True)
        )

    def _open_store(self, state: _StoreState) -> None:
        candidates = self._build_store_candidates(state.commands)
        candidate_keys = store_transfer_candidate_keys(candidates)
        accepted, claim_once = self._admitted_store_keys(candidate_keys)
        if accepted is not None and not accepted:
            return
        transfer_candidate_mask = (
            None
            if accepted is None
            else select_store_transfer_candidates(candidate_keys, accepted, claim_once=claim_once)
        )
        batch = self._materialize_store_candidates(state.commands, candidates, transfer_candidate_mask)
        state.batch = batch
        # Admission has already chosen one source owner per key, including duplicates across requests.
        keys: list[str] = []
        object_sizes: list[int] = []
        for group in batch.groups:
            group_keys = group.selected_keys()
            keys.extend(group_keys)
            object_sizes.extend([group.remote_object_size_bytes] * len(group_keys))
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
            state.plan = self._prepare_store_layers(selected)
            state.selected_keys_by_layer = state.plan.selected_keys_by_layer
        except Exception as error:
            state.failure_codes.update((key, None) for key in state.opened_keys)
            state.errors_by_key.update((key, error) for key in state.opened_keys)

    def _store_layer(self, source_ready: Any, layer_id: int, state: _StoreState) -> None:
        keys = state.selected_keys_by_layer.get(layer_id)
        if keys:
            assert state.plan is not None
            result = self._copy_store_layer(source_ready, layer_id, state.plan)
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
            state.plan = None
            self._complete_shared_store(state, [])
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
                self._record_store_result(
                    state, state.selected_keys_by_layer[layer_id], LayerStoreResult(None, False, error)
                )
        state.plan = None
        if state.batch is None:
            self._complete_shared_store(state, [])
            return _store_command_completions(state.commands)

        # Preparation succeeded and every submitted Future was observed; no second layer ledger is needed.
        missing_layers = state.selected_keys_by_layer.keys() - layers.keys()
        if missing_layers:
            incomplete = RuntimeError(f"Layerwise Store did not reach physical layers {sorted(missing_layers)}")
            for key in state.opened_keys:
                state.failure_codes.setdefault(key, None)
                state.errors_by_key.setdefault(key, incomplete)
        commit_keys = [key for key in state.opened_keys if key not in state.failure_codes]
        commit_keys = self._complete_shared_store(state, commit_keys)
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
        observed_layers = state.selected_keys_by_layer.keys() & layers.keys()
        return self._build_store_completions(state, observed_layers, final_codes)

    def _complete_shared_store(self, state: _StoreState, commit_keys: list[str]) -> list[str]:
        """Concrete shared-object routes require peer evidence before publication."""
        return commit_keys

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
                start = int(group.request_transfer_record_boundaries[request_index])
                end = int(group.request_transfer_record_boundaries[request_index + 1])
                for partition_index, axis in enumerate(group.keys_by_partition):
                    partition_candidate_start = partition_index * group.transfer_record_count
                    for transfer_record_index in range(start, end):
                        transfer_candidate_index = partition_candidate_start + transfer_record_index
                        if (
                            group.transfer_candidate_mask is not None
                            and not group.transfer_candidate_mask[transfer_candidate_index]
                        ):
                            continue
                        key = axis[transfer_record_index]
                        if key in state.failure_codes:
                            failed_requests.add(request_index)
                        if key in state.errors_by_key:
                            errors_by_request.setdefault(request_index, state.errors_by_key[key])
                        if key not in opened_keys or not physical_layer_ids:
                            continue
                        source = TransferSource(
                            group,
                            transfer_candidate_index,
                            transfer_record_index,
                            request_index,
                            physical_layer_ids,
                            key,
                        )
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

    def _seal_store(self, context: KVPoolStepContext | None = None) -> None:
        """Seal and register the current Store once, without waiting for its result.

        Keep active ownership and the sealed record until registration succeeds,
        including retries after only part of the pending bookkeeping completed.
        """
        state = self._store_state
        if state is None:
            return
        pending = state.pending
        if self._pending_store_batches and self._pending_store_batches[-1] is not pending:
            raise RuntimeError("Previous Store invocation has not reached its fence")
        if pending is None:
            assert self._store_opening is not None
            future = self._get_session().finish_store(self._store_opening, self._finish_store, state)
            pending = PendingStore(
                future,
                frozenset(command.request_id for command in state.commands),
                frozenset(command.store_job_id for command in state.commands if command.store_job_id is not None),
            )
            state.pending = pending
        self._track_store_batch(pending)
        # Queued tasks retain this invocation after the Step relinquishes its entry points.
        self._store_state = None
        self._store_opening = None
        if context is not None:
            context.store_submitted = True

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
        return layer_id in state.selected_keys_by_layer

    def _wait_for_layer_load(self, layer_name: str) -> tuple[LoadCompletion, ...]:
        layer_id = self._layer_ids_by_name.get(layer_name)
        if layer_id is None:
            return ()
        store_at_attention_entry = layer_id in self._attention_store_layers and self._can_store_layer(layer_id)
        if self._session is None and not store_at_attention_entry:
            return ()
        state = self._load_state
        try:
            session = self._get_session()
            result = None
            if state is not None and state.prepared is not None:
                result = session.wait_for_layer(layer_id, state.prepared.copy_layer, self._backend_sessions.finish_load)
            if store_at_attention_entry:
                session.bind_store_to_compute_start(layer_id, self.save_layer, layer_name)
            return result or ()
        except (BrokenExecutor, CancelledError):
            raise
        except Exception:
            if state is None:
                raise
            logger.exception("Layerwise Load failed at physical layer %d", layer_id)
            self._abort_load()
            return _failed_load_completions(state.batch, layer_id)

    def end_step(self) -> None:
        if self._session is not None or self._backend_sessions.has_load_sessions:
            finish_load = self._backend_sessions.finish_load if self._backend_sessions.has_load_sessions else None
            self._get_session().end_step(finish_load)
        self._seal_store(self._active_step)
        self._session = None
        self._load_state = None
        super().end_step()

    def close(self) -> None:
        if self._session is not None or self._backend_sessions.has_load_sessions:
            finish_load = self._backend_sessions.finish_load if self._backend_sessions.has_load_sessions else None
            self._get_session().end_step(finish_load, wait=False)
        super().close()

    def _check_execution_results(self) -> None:
        if self._session is not None:
            self._session.wait_load_finished()


def collect_object_sizes(batch: KVTransferBatch) -> dict[str, int]:
    object_sizes: dict[str, int] = {}
    for group in batch.groups:
        for key in group.selected_keys():
            previous = object_sizes.setdefault(key, group.remote_object_size_bytes)
            if previous != group.remote_object_size_bytes:
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
            range(group.transfer_candidate_count)
            if group.transfer_candidate_mask is None
            else np.flatnonzero(group.transfer_candidate_mask).tolist()
        )
        for transfer_candidate_index in selected:
            partition_index, transfer_record_index = divmod(transfer_candidate_index, group.transfer_record_count)
            code = failed_codes.get(group.keys_by_partition[partition_index][transfer_record_index], 0)
            if code != 0:
                source = group.source(transfer_candidate_index, None)
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


__all__ = ("LayerwiseWorker",)
