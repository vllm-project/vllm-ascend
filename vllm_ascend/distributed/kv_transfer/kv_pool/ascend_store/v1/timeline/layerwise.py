"""Coordinate layer-triggered transfers with session-wide visibility fences.

Load and Store sessions own Backend resources across multiple layer hooks.
Layerwise Store queues command preparation before layer work, reuses only the
admitted missing rows, and publishes request completion after every submitted
layer has reached the session's commit or revoke boundary.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from vllm.logger import logger

from ..protocol.transfer import StoreCommand
from ..runtime.batch import KVTransferBatch, TransferSource
from ..runtime.evidence import LayerStoreResult, LoadCompletion, StoreCompletion, StoreEvidence, TransferEvidence
from ..topology import KVPoolTopology
from . import LayerwiseStoreOperation, LayerwiseStorePreparation, LoadOperation, LoadTimelineProtocol, StoreBatch
from .executor import TimelineExecutor

if TYPE_CHECKING:
    from ...attention_fence import AttentionComputeStartGate

_TIMELINE_POLL_INTERVAL_S = 1.0


class LayerwiseBackendOperations(Protocol):
    def validate_support(self) -> None: ...

    def start_load_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]: ...

    def finish_load_sessions(self, keys: list[str]) -> None: ...

    def start_store_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]: ...

    def commit_store_sessions(self, keys: list[str]) -> tuple[int, ...]: ...

    def revoke_store_sessions(self, keys: list[str]) -> tuple[int, ...]: ...


class LayerwiseLoadTimelineProtocol(LoadTimelineProtocol, Protocol):
    def wait_for_layer(self, layer_name: str) -> Iterable[LoadCompletion]: ...


class LayerwiseStoreTimelineProtocol(Protocol):
    def bind_preparation(self, preparation: LayerwiseStorePreparation) -> None: ...

    def bind_operation(self, operation: LayerwiseStoreOperation) -> None: ...

    def start(self) -> None: ...

    def prepare(self, commands: tuple[StoreCommand, ...]) -> None: ...

    def submit_layer(self, layer_name: str, record_source_ready: Callable[[], Any]) -> None: ...

    def finalize(self) -> StoreBatch: ...

    def wait(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]: ...

    def prepare_close(self) -> StoreBatch | None: ...

    def close(self) -> StoreBatch | None: ...


@dataclass(slots=True)
class _OpenLoadSession:
    batch: KVTransferBatch
    keys: tuple[str, ...]
    object_sizes: tuple[int, ...]
    completed: threading.Event = field(default_factory=threading.Event)
    session: _LoadSession | None = None
    completions: tuple[LoadCompletion, ...] = ()


@dataclass(slots=True)
class _LoadLayerJob:
    layer_id: int
    session: _LoadSession
    start_gate: AttentionComputeStartGate | None = None
    completed: threading.Event = field(default_factory=threading.Event)
    completions: tuple[LoadCompletion, ...] = ()
    submitted: bool = False
    consumed: bool = False


@dataclass(slots=True)
class _LoadSession:
    batch: KVTransferBatch
    session_keys: tuple[str, ...]
    layer_order: tuple[int, ...]
    layer_jobs: dict[int, _LoadLayerJob] = field(default_factory=dict)
    next_layer_index: int = 0
    finish_scheduled: bool = False
    finish_completed: threading.Event = field(default_factory=threading.Event)
    session_end_attempted: bool = False
    session_end_confirmed: bool = False
    close_requested: bool = False


@dataclass(frozen=True, slots=True)
class _FinishLoadSession:
    session: _LoadSession


class LayerwiseLoadTimeline:
    """Prefetch layer ranges while preserving each visibility fence."""

    collects_completions = False

    def __init__(
        self,
        topology: KVPoolTopology,
        backend_io: LayerwiseBackendOperations,
        prefetch_layers: int,
        thread_initializer: Callable[[], None],
        start_gate_factory: Callable[[], AttentionComputeStartGate],
    ) -> None:
        if prefetch_layers <= 0:
            raise ValueError("Layerwise Load prefetch depth must be positive")
        self._backend_io = backend_io
        self._layer_ids_by_name = _compile_layer_ids_by_name(topology)
        self._layer_order = tuple(sorted(set(self._layer_ids_by_name.values())))
        self._prefetch_layers = prefetch_layers
        self._start_gate_factory = start_gate_factory
        self._operation: LoadOperation | None = None
        self._session: _LoadSession | None = None
        self._lifecycle_lock = threading.Lock()
        self._executor = TimelineExecutor(
            "KVPoolLayerwiseLoadExecutor", thread_initializer, self._execute, self._complete
        )

    def bind_operation(self, operation: LoadOperation) -> None:
        with self._lifecycle_lock:
            if self._operation is not None:
                raise RuntimeError("Layerwise Load operation is already bound")
            self._operation = operation

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._operation is None:
                raise RuntimeError("Layerwise Load timeline has no bound operation")
            self._backend_io.validate_support()
        self._executor.start()
        self._raise_if_failed()

    def submit(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            if self._session is not None:
                raise RuntimeError("Previous Layerwise Load session has not completed")
            object_sizes = _collect_object_sizes(batch)
            command = _OpenLoadSession(batch, tuple(object_sizes), tuple(object_sizes.values()))
            self._executor.submit(command)
        self._wait_for_completion(command.completed)
        with self._lifecycle_lock:
            self._session = command.session
        return command.completions

    def wait_for_layer(self, layer_name: str) -> tuple[LoadCompletion, ...]:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            session = self._session
            if session is None:
                return ()
            layer_id = self._layer_ids_by_name.get(layer_name)
            if layer_id is None:
                return ()
            layer_job = session.layer_jobs.get(layer_id)
            if layer_job is not None and layer_job.consumed:
                return ()
            start_gate = self._start_gate_factory()
            self._fill_prefetch_window(session, layer_id, start_gate)
            if layer_job is None:
                return ()
            self._submit_through_layer(session, layer_id, start_gate)
        self._wait_for_completion(layer_job.completed)
        with self._lifecycle_lock:
            layer_job.consumed = True
            completions = layer_job.completions
            all_layers_consumed = all(item.consumed for item in session.layer_jobs.values())
        if all_layers_consumed:
            self._wait_for_finish(session)
            with self._lifecycle_lock:
                if self._session is session:
                    self._session = None
        return completions

    def collect(self) -> tuple[LoadCompletion, ...]:
        self._raise_if_failed()
        with self._lifecycle_lock:
            session = self._session
            if session is None:
                return ()
            pending_layers = sorted(
                layer_id for layer_id, layer_job in session.layer_jobs.items() if not layer_job.consumed
            )
            self._request_finish(session)
        self._wait_for_finish(session)
        error = RuntimeError(f"Layerwise Load did not reach physical layers {pending_layers}")
        self._executor.terminate(error)
        with self._lifecycle_lock:
            if self._session is session:
                self._session = None
        raise RuntimeError("Layerwise Load timeline terminated") from error

    def abort(self) -> None:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            session = self._session
            if session is None:
                return
            self._request_finish(session)
        self._wait_for_finish(session)
        with self._lifecycle_lock:
            if self._session is session:
                self._session = None

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._executor.closed:
                return
            session = self._session
            if session is not None:
                self._request_finish(session)
        if session is not None:
            self._wait_for_finish(session)
        with self._lifecycle_lock:
            self._session = None
        self._executor.close()
        self._raise_if_failed()

    def _execute(self, command: _OpenLoadSession | _LoadLayerJob | _FinishLoadSession) -> None:
        try:
            if isinstance(command, _OpenLoadSession):
                self._open_session(command)
            elif isinstance(command, _LoadLayerJob):
                self._execute_layer(command)
            else:
                self._finish_session(command.session)
        except BaseException:
            keys = command.keys if isinstance(command, _OpenLoadSession) else ()
            self._cleanup_after_failure(keys)
            raise

    @staticmethod
    def _complete(command: _OpenLoadSession | _LoadLayerJob | _FinishLoadSession) -> None:
        if isinstance(command, _FinishLoadSession):
            command.session.finish_completed.set()
        else:
            command.completed.set()

    def _open_session(self, command: _OpenLoadSession) -> None:
        result_codes = self._backend_io.start_load_sessions(list(command.keys), list(command.object_sizes))
        codes_by_key = dict(zip(command.keys, result_codes, strict=True))
        session_keys = tuple(key for key, code in codes_by_key.items() if code == 0)
        command.completions = _load_session_failures(command.batch, codes_by_key)
        selected = command.batch.select_keys(set(session_keys))
        layer_order = tuple(
            layer_id
            for layer_id in self._layer_order
            if any(layer_id in group.physical_layer_ids and group.selected_keys() for group in selected.groups)
        )
        if not layer_order:
            self._backend_io.finish_load_sessions(list(session_keys))
            return
        session = _LoadSession(selected, session_keys, layer_order)
        session.layer_jobs = {layer_id: _LoadLayerJob(layer_id, session) for layer_id in layer_order}
        command.session = session

    def _execute_layer(self, job: _LoadLayerJob) -> None:
        assert self._operation is not None
        if job.start_gate is not None:
            while not job.start_gate.wait(timeout=10):
                logger.info("Layerwise %d load waits for attention compute start", job.layer_id)
        if job.session.close_requested:
            return
        job.completions = self._operation(job.session.batch.for_layer(job.layer_id), job.layer_id)

    def _fill_prefetch_window(
        self, session: _LoadSession, current_layer_id: int, start_gate: AttentionComputeStartGate
    ) -> None:
        active = sum(item.submitted and not item.consumed for item in session.layer_jobs.values())
        while active < self._prefetch_layers and session.next_layer_index < len(session.layer_order):
            layer_id = session.layer_order[session.next_layer_index]
            session.next_layer_index += 1
            job = session.layer_jobs[layer_id]
            if not job.submitted:
                job.start_gate = None if layer_id == current_layer_id else start_gate
                job.submitted = True
                self._executor.submit(job)
                active += 1
        self._schedule_finish_if_complete(session)

    def _submit_through_layer(
        self, session: _LoadSession, layer_id: int, start_gate: AttentionComputeStartGate
    ) -> None:
        target_index = session.layer_order.index(layer_id)
        while session.next_layer_index <= target_index:
            next_layer_id = session.layer_order[session.next_layer_index]
            session.next_layer_index += 1
            job = session.layer_jobs[next_layer_id]
            if not job.submitted:
                job.start_gate = None if next_layer_id == layer_id else start_gate
                job.submitted = True
                self._executor.submit(job)
        self._schedule_finish_if_complete(session)

    def _request_finish(self, session: _LoadSession) -> None:
        session.close_requested = True
        for job in session.layer_jobs.values():
            if job.submitted and not job.completed.is_set() and job.start_gate is not None:
                job.start_gate.cancel()
        self._schedule_finish(session)

    def _schedule_finish_if_complete(self, session: _LoadSession) -> None:
        if session.next_layer_index == len(session.layer_order):
            self._schedule_finish(session)

    def _schedule_finish(self, session: _LoadSession) -> None:
        if session.finish_scheduled or session.session_end_confirmed:
            return
        session.finish_scheduled = True
        self._executor.submit(_FinishLoadSession(session))

    def _wait_for_completion(self, completed: threading.Event) -> None:
        while True:
            self._raise_if_failed()
            if completed.wait(timeout=_TIMELINE_POLL_INTERVAL_S):
                self._raise_if_failed()
                return

    def _wait_for_finish(self, session: _LoadSession) -> None:
        self._wait_for_completion(session.finish_completed)

    def _finish_session(self, session: _LoadSession) -> None:
        if session.session_end_confirmed:
            return
        if session.session_end_attempted:
            raise RuntimeError("Layerwise Load session end was attempted but not confirmed")
        session.session_end_attempted = True
        self._backend_io.finish_load_sessions(list(session.session_keys))
        session.session_end_confirmed = True

    def _raise_if_failed(self) -> None:
        if self._executor.failure is not None:
            raise RuntimeError("Layerwise Load timeline terminated") from self._executor.failure

    def _raise_if_not_running(self) -> None:
        self._raise_if_failed()
        self._executor.check_running()

    def _cleanup_after_failure(self, keys: tuple[str, ...] = ()) -> None:
        session = self._session
        cleanup_keys = keys if session is None else session.session_keys
        should_close = bool(cleanup_keys) and (session is None or not session.session_end_attempted)
        if should_close:
            try:
                if session is not None:
                    session.session_end_attempted = True
                self._backend_io.finish_load_sessions(list(cleanup_keys))
                if session is not None:
                    session.session_end_confirmed = True
            except BaseException:
                logger.exception("Failed to close Layerwise Load sessions after an execution error")


@dataclass(slots=True)
class _StoreSession:
    commands: tuple[StoreCommand, ...]
    batch: KVTransferBatch | None = None
    terminal_completions: tuple[StoreCompletion, ...] = ()
    pending_finalization_keys: tuple[str, ...] = ()
    selected: KVTransferBatch | None = None
    request_keys: tuple[frozenset[str], ...] = ()
    observed_layer_ids: set[int] = field(default_factory=set)
    pending_layer_ids: set[int] = field(default_factory=set)
    submitted_layer_ids: set[int] = field(default_factory=set)
    failed_keys: set[str] = field(default_factory=set)
    session_result_codes: dict[str, int | None] = field(default_factory=dict)
    range_failure_codes: dict[str, int | None] = field(default_factory=dict)
    unsafe_keys: set[str] = field(default_factory=set)
    errors_by_key: dict[str, Exception] = field(default_factory=dict)

    def open(
        self,
        batch: KVTransferBatch | None,
        terminal_completions: tuple[StoreCompletion, ...],
        backend_io: LayerwiseBackendOperations,
    ) -> None:
        self.batch = batch
        self.terminal_completions = terminal_completions
        if batch is None:
            return
        self.request_keys = _keys_by_request(batch)
        object_sizes_by_key = _collect_object_sizes(batch)
        selected_keys = batch.selected_keys()
        session_keys = tuple(dict.fromkeys(selected_keys))
        if session_keys:
            self._start_sessions(session_keys, object_sizes_by_key, backend_io)
        self.selected = (
            batch
            if self.pending_finalization_keys == selected_keys
            else batch.select_keys(set(self.pending_finalization_keys), claim_once=True)
        )
        self.pending_layer_ids = {
            layer_id for group in self.selected.groups if group.selected_keys() for layer_id in group.physical_layer_ids
        }

    def record_range(self, result: LayerStoreResult, layer_id: int, expected_keys: tuple[str, ...]) -> None:
        if self.batch is None:
            raise RuntimeError("Layerwise Store range completed without a prepared batch")
        if not (len(result.keys) == len(result.result_codes) == len(result.source_release_confirmed)):
            raise RuntimeError("Layerwise Store result axes are not aligned")
        if result.keys != expected_keys:
            raise RuntimeError(f"Layerwise Store layer {layer_id} returned results for unexpected keys")
        self.observed_layer_ids.add(layer_id)
        for key, code, released in zip(result.keys, result.result_codes, result.source_release_confirmed, strict=True):
            if code != 0:
                self.failed_keys.add(key)
                self.range_failure_codes.setdefault(key, code)
            if not released:
                self.unsafe_keys.add(key)
            if result.error is not None:
                self.errors_by_key.setdefault(key, result.error)

    def finalize(self, backend_io: LayerwiseBackendOperations) -> tuple[StoreCompletion, ...]:
        if self.terminal_completions:
            return self.terminal_completions
        if self.batch is None:
            raise RuntimeError("Layerwise Store preparation produced neither a batch nor completions")
        incomplete_layer_ids = tuple(sorted(self.pending_layer_ids))
        if incomplete_layer_ids:
            error = RuntimeError(f"Layerwise Store did not reach physical layers {list(incomplete_layer_ids)}")
            for key in self.pending_finalization_keys:
                self.failed_keys.add(key)
                self.errors_by_key.setdefault(key, error)
        commit_keys = [key for key in self.pending_finalization_keys if key not in self.failed_keys]
        if commit_keys:
            try:
                result_codes = backend_io.commit_store_sessions(commit_keys)
            except Exception as error:
                self.failed_keys.update(commit_keys)
                self.errors_by_key.update((key, error) for key in commit_keys)
            else:
                self._record_session_results(commit_keys, result_codes)
        revoke_keys = [key for key in self.pending_finalization_keys if key in self.failed_keys]
        if revoke_keys:
            self._revoke_sessions(revoke_keys, backend_io)
        return self._build_completions()

    def revoke_pending(self, backend_io: LayerwiseBackendOperations) -> None:
        if not self.pending_finalization_keys:
            return
        try:
            backend_io.revoke_store_sessions(list(self.pending_finalization_keys))
        except BaseException:
            logger.exception("Failed to revoke Layerwise Store sessions after a timeline error")

    def _start_sessions(
        self, missing_keys: tuple[str, ...], object_sizes_by_key: dict[str, int], backend_io: LayerwiseBackendOperations
    ) -> None:
        try:
            result_codes = backend_io.start_store_sessions(
                list(missing_keys), [object_sizes_by_key[key] for key in missing_keys]
            )
        except Exception as error:
            self.failed_keys.update(missing_keys)
            self.errors_by_key.update((key, error) for key in missing_keys)
            self._revoke_sessions(list(missing_keys), backend_io)
            return
        self.pending_finalization_keys = tuple(
            key for key, code in zip(missing_keys, result_codes, strict=True) if code == 0
        )
        for key, code in zip(missing_keys, result_codes, strict=True):
            if code != 0:
                self.failed_keys.add(key)
                self.session_result_codes[key] = code

    def _record_session_results(self, keys: list[str], result_codes: tuple[int, ...]) -> None:
        for key, code in zip(keys, result_codes, strict=True):
            self.session_result_codes[key] = code
            if code != 0:
                self.failed_keys.add(key)
        committed = {key for key, code in zip(keys, result_codes, strict=True) if code == 0}
        self.pending_finalization_keys = tuple(key for key in self.pending_finalization_keys if key not in committed)

    def _revoke_sessions(self, keys: list[str], backend_io: LayerwiseBackendOperations) -> None:
        try:
            result_codes = backend_io.revoke_store_sessions(keys)
        except Exception as error:
            for key in keys:
                self.errors_by_key.setdefault(key, error)
        else:
            for key, code in zip(keys, result_codes, strict=True):
                if code != 0:
                    self.session_result_codes.setdefault(key, code)
                    self.errors_by_key.setdefault(
                        key, RuntimeError(f"Store session revocation was not confirmed; result code {code}")
                    )
        self.pending_finalization_keys = tuple(key for key in self.pending_finalization_keys if key not in keys)

    def _build_completions(self) -> tuple[StoreCompletion, ...]:
        if self.batch is None:
            raise RuntimeError("Layerwise Store cannot build completions without a prepared batch")
        batch = self.batch
        job_ids = batch.store_job_ids or (None,) * len(batch.request_ids)
        selected = self.selected
        if selected is None:
            raise RuntimeError("Layerwise Store cannot build completions without selected sessions")
        sources_by_request = _sources_by_request(selected, self.observed_layer_ids)
        completions = []
        for request_index, (request_id, store_job_id) in enumerate(zip(batch.request_ids, job_ids, strict=True)):
            evidence = tuple(
                TransferEvidence(
                    source,
                    self.session_result_codes.get(source.key, self.range_failure_codes.get(source.key)),
                    source.key not in self.unsafe_keys,
                )
                for source in sources_by_request[request_index]
            )
            keys = self.request_keys[request_index]
            failed = keys & self.failed_keys
            error = next((self.errors_by_key[key] for key in keys if key in self.errors_by_key), None)
            source_release_confirmed = all(item.source_release_confirmed is True for item in evidence)
            completions.append(
                StoreCompletion(
                    request_id,
                    StoreEvidence(evidence, not failed and error is None, source_release_confirmed, error),
                    store_job_id,
                )
            )
        return tuple(completions)


@dataclass(slots=True)
class _OpenStoreSession:
    session: _StoreSession


@dataclass(frozen=True, slots=True)
class _StoreLayerJob:
    session: _StoreSession
    layer_id: int
    source_ready_event: Any


@dataclass(frozen=True, slots=True)
class _FinalizeStoreSession:
    session: _StoreSession
    batch: StoreBatch


class LayerwiseStoreTimeline:
    """Publish layer ranges, then commit or revoke their shared objects."""

    def __init__(
        self, topology: KVPoolTopology, backend_io: LayerwiseBackendOperations, thread_initializer: Callable[[], None]
    ) -> None:
        self._backend_io = backend_io
        self._layer_ids_by_name = _compile_layer_ids_by_name(topology)
        self._preparation: LayerwiseStorePreparation | None = None
        self._operation: LayerwiseStoreOperation | None = None
        self._lifecycle_lock = threading.Lock()
        self._session: _StoreSession | None = None
        self._pending_batch: StoreBatch | None = None
        self._executor = TimelineExecutor(
            "KVPoolLayerwiseStoreExecutor", thread_initializer, self._execute, self._complete
        )

    def bind_preparation(self, preparation: LayerwiseStorePreparation) -> None:
        with self._lifecycle_lock:
            if self._preparation is not None:
                raise RuntimeError("Layerwise Store preparation is already bound")
            self._preparation = preparation

    def bind_operation(self, operation: LayerwiseStoreOperation) -> None:
        with self._lifecycle_lock:
            if self._operation is not None:
                raise RuntimeError("Layerwise Store operation is already bound")
            self._operation = operation

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._preparation is None or self._operation is None:
                raise RuntimeError("Layerwise Store timeline is not fully bound")
            self._backend_io.validate_support()
        self._executor.start()
        self._raise_if_failed()

    def prepare(self, commands: tuple[StoreCommand, ...]) -> None:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            if self._session is not None or self._pending_batch is not None:
                raise RuntimeError("Previous Layerwise Store session has not reached its fence")
            session = _StoreSession(commands)
            self._session = session
            self._executor.submit(_OpenStoreSession(session))

    def submit_layer(self, layer_name: str, record_source_ready: Callable[[], Any]) -> None:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            session = self._session
            if session is None:
                return
            layer_id = self._layer_ids_by_name.get(layer_name)
            if layer_id is None or layer_id in session.submitted_layer_ids:
                return
            source_ready_event = record_source_ready()
            session.submitted_layer_ids.add(layer_id)
            self._executor.submit(_StoreLayerJob(session, layer_id, source_ready_event))

    def finalize(self) -> StoreBatch:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            return self._finalize_session()

    def _finalize_session(self) -> StoreBatch:
        session = self._session
        if session is None:
            raise RuntimeError("Layerwise Store session has not been prepared")
        batch = StoreBatch()
        self._session = None
        self._pending_batch = batch
        self._executor.submit(_FinalizeStoreSession(session, batch))
        return batch

    def wait(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]:
        while True:
            self._raise_if_failed()
            if batch.completed.wait(timeout=_TIMELINE_POLL_INTERVAL_S):
                break
        self._raise_if_failed()
        with self._lifecycle_lock:
            if self._pending_batch is batch:
                self._pending_batch = None
        return tuple(batch.completions)

    def prepare_close(self) -> StoreBatch | None:
        with self._lifecycle_lock:
            if self._session is None:
                return self._pending_batch
            self._raise_if_not_running()
            return self._finalize_session()

    def close(self) -> StoreBatch | None:
        batch = None
        try:
            batch = self.prepare_close()
            if batch is not None:
                self.wait(batch)
        finally:
            self._executor.close()
        self._raise_if_failed()
        return batch

    def _execute(self, command: _OpenStoreSession | _StoreLayerJob | _FinalizeStoreSession) -> None:
        try:
            if isinstance(command, _OpenStoreSession):
                assert self._preparation is not None
                batch, completions = self._preparation(command.session.commands)
                command.session.open(batch, completions, self._backend_io)
            elif isinstance(command, _StoreLayerJob):
                self._execute_layer(command)
            else:
                command.batch.completions.extend(command.session.finalize(self._backend_io))
        except BaseException:
            command.session.revoke_pending(self._backend_io)
            raise

    @staticmethod
    def _complete(command: _OpenStoreSession | _StoreLayerJob | _FinalizeStoreSession) -> None:
        if isinstance(command, _FinalizeStoreSession):
            command.batch.completed.set()

    def _execute_layer(self, job: _StoreLayerJob) -> None:
        assert self._operation is not None
        if job.layer_id not in job.session.pending_layer_ids:
            return
        job.session.pending_layer_ids.remove(job.layer_id)
        if job.session.selected is None:
            raise RuntimeError("Layerwise Store layer executed without a prepared batch")
        layer_batch = job.session.selected.for_layer(job.layer_id)
        result = self._operation(layer_batch, job.source_ready_event, job.layer_id)
        job.session.record_range(result, job.layer_id, layer_batch.selected_keys())

    def _raise_if_failed(self) -> None:
        if self._executor.failure is not None:
            raise RuntimeError("Layerwise Store failed during asynchronous Store") from self._executor.failure

    def _raise_if_not_running(self) -> None:
        self._raise_if_failed()
        self._executor.check_running()


def _collect_object_sizes(batch: KVTransferBatch) -> dict[str, int]:
    object_sizes: dict[str, int] = {}
    for group in batch.groups:
        for key in group.selected_keys():
            previous = object_sizes.setdefault(key, group.object_size)
            if previous != group.object_size:
                raise ValueError(f"Layerwise key {key!r} has inconsistent object sizes")
    return object_sizes


def _load_session_failures(batch: KVTransferBatch, codes_by_key: dict[str, int]) -> tuple[LoadCompletion, ...]:
    evidence_by_request: list[list[TransferEvidence]] = [[] for _ in batch.request_ids]
    for group in batch.groups:
        selected = range(group.object_count) if group.selection is None else np.flatnonzero(group.selection).tolist()
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


def _keys_by_request(batch: KVTransferBatch) -> tuple[frozenset[str], ...]:
    keys_by_request: list[set[str]] = [set() for _ in batch.request_ids]
    for group in batch.groups:
        for request_index, keys in enumerate(keys_by_request):
            start = int(group.request_splits[request_index])
            end = int(group.request_splits[request_index + 1])
            for axis_index, axis in enumerate(group.key_axes):
                axis_offset = axis_index * group.row_count
                for row_index in range(start, end):
                    object_index = axis_offset + row_index
                    if group.selection is None or group.selection[object_index]:
                        keys.add(axis[row_index])
    return tuple(frozenset(keys) for keys in keys_by_request)


def _sources_by_request(batch: KVTransferBatch, observed_layer_ids: set[int]) -> tuple[tuple[TransferSource, ...], ...]:
    sources_by_request: list[list[TransferSource]] = [[] for _ in batch.request_ids]
    for group in batch.groups:
        physical_layer_ids = tuple(layer_id for layer_id in group.physical_layer_ids if layer_id in observed_layer_ids)
        if not physical_layer_ids:
            continue
        for request_index, sources in enumerate(sources_by_request):
            start = int(group.request_splits[request_index])
            end = int(group.request_splits[request_index + 1])
            for axis_index, axis in enumerate(group.key_axes):
                axis_offset = axis_index * group.row_count
                for row_index in range(start, end):
                    object_index = axis_offset + row_index
                    if group.selection is None or group.selection[object_index]:
                        source = TransferSource(
                            group, object_index, row_index, request_index, physical_layer_ids, axis[row_index]
                        )
                        sources.append(source)
    return tuple(tuple(sources) for sources in sources_by_request)


def _compile_layer_ids_by_name(topology: KVPoolTopology) -> dict[str, int]:
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
