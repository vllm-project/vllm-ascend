"""Run Layerwise sessions while reusing one request batch across all layers."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from vllm.logger import logger

from ...program.spec.topology import KVPoolTopology
from ..batch import KVTransferBatch
from ..evidence import LoadCompletion, StoreCompletion, StoreEvidence, TransferEvidence
from . import LoadTimelineProtocol, StoreBatch
from .executor import TimelineExecutor

if TYPE_CHECKING:
    from ....attention_fence import AttentionComputeStartGate

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
    def bind_admission(self, admission: Callable[[KVTransferBatch], KVTransferBatch]) -> None: ...

    def bind_operation(
        self,
        operation: Callable[[KVTransferBatch, Any, int | None], tuple[StoreCompletion, ...]],
    ) -> None: ...

    def start(self) -> None: ...

    def prepare(self, batch: KVTransferBatch) -> None: ...

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
        self._operation: Callable[[KVTransferBatch, int | None], tuple[LoadCompletion, ...]] | None = None
        self._session: _LoadSession | None = None
        self._lifecycle_lock = threading.Lock()
        self._executor = TimelineExecutor(
            "KVPoolLayerwiseLoadExecutor", thread_initializer, self._execute, self._complete
        )

    def bind_operation(
        self,
        operation: Callable[[KVTransferBatch, int | None], tuple[LoadCompletion, ...]],
    ) -> None:
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
        self,
        session: _LoadSession,
        current_layer_id: int,
        start_gate: AttentionComputeStartGate,
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
        self,
        session: _LoadSession,
        layer_id: int,
        start_gate: AttentionComputeStartGate,
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
    batch: KVTransferBatch
    request_indices: dict[str, int]
    existing_keys: set[str] = field(default_factory=set)
    pending_finalization_keys: tuple[str, ...] = ()
    selected: KVTransferBatch | None = None
    pending_layer_ids: set[int] = field(default_factory=set)
    failed_keys: set[str] = field(default_factory=set)
    session_result_codes: dict[str, int | None] = field(default_factory=dict)
    range_result_codes: dict[tuple[str, int], int | None] = field(default_factory=dict)
    evidence_by_request: list[list[TransferEvidence]] = field(default_factory=list)
    unsafe_requests: set[int] = field(default_factory=set)
    errors_by_key: dict[str, Exception] = field(default_factory=dict)

    def open(
        self,
        admitted: KVTransferBatch,
        object_sizes_by_key: dict[str, int],
        backend_io: LayerwiseBackendOperations,
    ) -> None:
        self.evidence_by_request = [[] for _ in self.batch.request_ids]
        missing_keys = tuple(dict.fromkeys(admitted.selected_keys()))
        self.existing_keys.update(set(object_sizes_by_key) - set(missing_keys))
        if missing_keys:
            self._start_sessions(missing_keys, object_sizes_by_key, backend_io)
        self.selected = admitted.select_keys(set(self.pending_finalization_keys), claim_once=True)
        self.pending_layer_ids = {
            layer_id for group in self.selected.groups if group.selected_keys() for layer_id in group.physical_layer_ids
        }

    def record_range(self, completions: tuple[StoreCompletion, ...], layer_id: int) -> None:
        for completion in completions:
            request_index = self.request_indices[completion.request_id]
            observed_keys = set()
            for item in completion.evidence.transfer_evidence:
                self.evidence_by_request[request_index].append(item)
                observed_keys.add(item.source.key)
                self.range_result_codes[item.source.key, layer_id] = item.result_code
                if item.result_code != 0:
                    self.failed_keys.add(item.source.key)
                if item.source_release_confirmed is not True:
                    self.unsafe_requests.add(request_index)
            if completion.evidence.error is not None:
                error_keys = observed_keys or _request_keys(self.selected or self.batch, request_index)
                for key in error_keys:
                    self.errors_by_key.setdefault(key, completion.evidence.error)
            if not completion.evidence.succeeded:
                self.failed_keys.update(observed_keys)

    def finalize(
        self,
        backend_io: LayerwiseBackendOperations,
        incomplete_layer_ids: tuple[int, ...],
    ) -> tuple[StoreCompletion, ...]:
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
        self,
        missing_keys: tuple[str, ...],
        object_sizes_by_key: dict[str, int],
        backend_io: LayerwiseBackendOperations,
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
                        key,
                        RuntimeError(f"Store session revocation was not confirmed; result code {code}"),
                    )
        self.pending_finalization_keys = tuple(key for key in self.pending_finalization_keys if key not in keys)

    def _build_completions(self) -> tuple[StoreCompletion, ...]:
        job_ids = self.batch.store_job_ids or (None,) * len(self.batch.request_ids)
        completions = []
        for request_index, (request_id, store_job_id) in enumerate(zip(self.batch.request_ids, job_ids, strict=True)):
            evidence = tuple(
                TransferEvidence(
                    item.source,
                    self.session_result_codes.get(
                        item.source.key,
                        self.range_result_codes.get((item.source.key, item.source.physical_layer_ids[0])),
                    ),
                    item.source_release_confirmed,
                )
                for item in self.evidence_by_request[request_index]
                if item.source.key not in self.existing_keys
            )
            keys = _request_keys(self.batch, request_index)
            failed = keys & self.failed_keys
            error = next((self.errors_by_key[key] for key in keys if key in self.errors_by_key), None)
            source_release_confirmed = request_index not in self.unsafe_requests and all(
                item.source_release_confirmed is True for item in evidence
            )
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
    batch: KVTransferBatch
    object_sizes_by_key: dict[str, int]
    completed: threading.Event = field(default_factory=threading.Event)
    session: _StoreSession | None = None


@dataclass(frozen=True, slots=True)
class _StoreLayerJob:
    session: _StoreSession
    layer_id: int
    source_ready_event: Any


@dataclass(frozen=True, slots=True)
class _FinalizeStoreSession:
    session: _StoreSession
    batch: StoreBatch
    incomplete_layer_ids: tuple[int, ...]


class LayerwiseStoreTimeline:
    """Publish layer ranges, then commit or revoke their shared objects."""

    def __init__(
        self,
        topology: KVPoolTopology,
        backend_io: LayerwiseBackendOperations,
        thread_initializer: Callable[[], None],
    ) -> None:
        self._backend_io = backend_io
        self._layer_ids_by_name = _compile_layer_ids_by_name(topology)
        self._admission: Callable[[KVTransferBatch], KVTransferBatch] | None = None
        self._operation: Callable[[KVTransferBatch, Any, int | None], tuple[StoreCompletion, ...]] | None = None
        self._lifecycle_lock = threading.Lock()
        self._session: _StoreSession | None = None
        self._pending_batch: StoreBatch | None = None
        self._executor = TimelineExecutor(
            "KVPoolLayerwiseStoreExecutor", thread_initializer, self._execute, self._complete
        )

    def bind_admission(self, admission: Callable[[KVTransferBatch], KVTransferBatch]) -> None:
        with self._lifecycle_lock:
            if self._admission is not None:
                raise RuntimeError("Layerwise Store admission is already bound")
            self._admission = admission

    def bind_operation(
        self,
        operation: Callable[[KVTransferBatch, Any, int | None], tuple[StoreCompletion, ...]],
    ) -> None:
        with self._lifecycle_lock:
            if self._operation is not None:
                raise RuntimeError("Layerwise Store operation is already bound")
            self._operation = operation

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._admission is None or self._operation is None:
                raise RuntimeError("Layerwise Store timeline is not fully bound")
            self._backend_io.validate_support()
        self._executor.start()
        self._raise_if_failed()

    def prepare(self, batch: KVTransferBatch) -> None:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            if self._session is not None or self._pending_batch is not None:
                raise RuntimeError("Previous Layerwise Store session has not reached its fence")
            self._session = self._open_session(batch)

    def submit_layer(self, layer_name: str, record_source_ready: Callable[[], Any]) -> None:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            session = self._session
            if session is None:
                return
            layer_id = self._layer_ids_by_name.get(layer_name)
            if layer_id is None or layer_id not in session.pending_layer_ids:
                return
            source_ready_event = record_source_ready()
            session.pending_layer_ids.remove(layer_id)
            self._executor.submit(_StoreLayerJob(session, layer_id, source_ready_event))

    def finalize(self) -> StoreBatch:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            return self._finalize_session()

    def _finalize_session(self) -> StoreBatch:
        session = self._session
        if session is None:
            raise RuntimeError("Layerwise Store session has not been prepared")
        batch = StoreBatch(session.batch)
        self._session = None
        self._pending_batch = batch
        incomplete = tuple(sorted(session.pending_layer_ids))
        self._executor.submit(_FinalizeStoreSession(session, batch, incomplete))
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
                assert self._admission is not None
                session = _StoreSession(
                    command.batch,
                    {request_id: index for index, request_id in enumerate(command.batch.request_ids)},
                )
                command.session = session
                admitted = self._admission(command.batch)
                session.open(admitted, command.object_sizes_by_key, self._backend_io)
            elif isinstance(command, _StoreLayerJob):
                self._execute_layer(command)
            else:
                command.batch.completions.extend(
                    command.session.finalize(self._backend_io, command.incomplete_layer_ids)
                )
        except BaseException:
            failed_session: _StoreSession | None = command.session
            if failed_session is not None:
                failed_session.revoke_pending(self._backend_io)
            raise

    @staticmethod
    def _complete(command: _OpenStoreSession | _StoreLayerJob | _FinalizeStoreSession) -> None:
        if isinstance(command, _OpenStoreSession):
            command.completed.set()
        elif isinstance(command, _FinalizeStoreSession):
            command.batch.completed.set()

    def _open_session(self, batch: KVTransferBatch) -> _StoreSession:
        command = _OpenStoreSession(batch, _collect_object_sizes(batch))
        self._executor.submit(command)
        self._wait_for_completion(command.completed)
        if command.session is None:
            raise RuntimeError("Layerwise Store executor did not create a session")
        return command.session

    def _execute_layer(self, job: _StoreLayerJob) -> None:
        assert self._operation is not None
        assert job.session.selected is not None
        layer_batch = job.session.selected.for_layer(job.layer_id)
        completions = self._operation(layer_batch, job.source_ready_event, job.layer_id)
        job.session.record_range(completions, job.layer_id)

    def _wait_for_completion(self, completed: threading.Event) -> None:
        while True:
            self._raise_if_failed()
            if completed.wait(timeout=_TIMELINE_POLL_INTERVAL_S):
                self._raise_if_failed()
                return

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


def _load_session_failures(
    batch: KVTransferBatch,
    codes_by_key: dict[str, int],
) -> tuple[LoadCompletion, ...]:
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


def _request_keys(batch: KVTransferBatch, request_index: int) -> set[str]:
    keys = set()
    for group in batch.groups:
        start = int(group.request_splits[request_index])
        end = int(group.request_splits[request_index + 1])
        for axis_index, axis in enumerate(group.key_axes):
            axis_offset = axis_index * group.row_count
            for row_index in range(start, end):
                object_index = axis_offset + row_index
                if group.selection is None or group.selection[object_index]:
                    keys.add(axis[row_index])
    return keys


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
