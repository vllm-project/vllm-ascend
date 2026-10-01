"""Advance pre-enumerated layer work through Backend range sessions.

Load exposes each completed range at its layer fence. Store accumulates layer
ranges and publishes the remote object only when the step is finalized::

    Layerwise Load: layer path                          Layerwise Load: abort path

    ┌──────────────────────────┐                        ┌──────────────────────────┐
    │ idle                     │                        │ session active           │
    └────────────┬─────────────┘                        └────────────┬─────────────┘
                 │ submit()                                          │ abort() / close()
                 │ start_load_sessions()                             ▼
                 ▼                                      ┌──────────────────────────┐
    ┌──────────────────────────┐                        │ close requested          │
    │ session active           │                        │ gated jobs canceled      │
    │ layer jobs pending       │                        └────────────┬─────────────┘
    └────────────┬─────────────┘                                     │ finish_load_sessions()
                 │ wait_for_layer()                                  ▼
                 ▼                                      ┌──────────────────────────┐
    ┌──────────────────────────┐                        │ end attempted            │
    │ layer job queued         │                        └────────────┬─────────────┘
    └────────────┬─────────────┘                                     ├── confirmed → idle
                 │ gate opens                                        └── failed → timeline failed
                 │ load_session_ranges()                                 end remains unconfirmed
                 ▼
    ┌──────────────────────────┐
    │ range complete           │
    └────────────┬─────────────┘
                 │ [last job]
                 │ finish_load_sessions()
                 │ hook consumes completion
                 ▼
    ┌──────────────────────────┐
    │ layer visible            │
    └────────────┬─────────────┘
                 ├── more layers → session active
                 └── all visible + end confirmed → idle

    Layerwise Store: layer path                         Layerwise Store: finalization path

    ┌──────────────────────────┐                        ┌──────────────────────────┐
    │ idle                     │                        │ session active           │
    └────────────┬─────────────┘                        └────────────┬─────────────┘
                 │ prepare()                                         │ submit() / close()
                 │ start_store_sessions()                            ▼
                 ▼                                      ┌──────────────────────────┐
    ┌──────────────────────────┐                        │ finalization queued      │
    │ session active           │                        └────────────┬─────────────┘
    │ layer ranges pending     │                                     │ evaluate object evidence
    └────────────┬─────────────┘                                     ▼
                 │ submit_layer()                       ┌──────────────────────────┐
                 │ source ready                         │ finalizing objects       │
                 ▼                                      │ clean keys: commit       │
    ┌──────────────────────────┐                        │ failed keys: revoke      │
    │ layer range queued       │                        └────────────┬─────────────┘
    └────────────┬─────────────┘                                     │ commit failure joins revoke set
                 │ store_session_ranges()                            │ aggregate all evidence
                 ▼                                                   ▼
    ┌──────────────────────────┐                        ┌──────────────────────────┐
    │ range evidence recorded  │                        │ completion ready         │
    └────────────┬─────────────┘                        └────────────┬─────────────┘
                 ├── next layer → session active                     │ signal StoreBatch / wait()
                 └── step end → finalization path                    ▼
                                                        ┌──────────────────────────┐
                                                        │ idle                     │
                                                        └──────────────────────────┘
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from vllm.logger import logger

from ...program.spec.topology import KVPoolTopology
from ...program.values.evidence import StoreEvidence, TransferEvidence
from ...program.values.selection import (
    merge_transfer_work,
    select_work_keys,
    selected_work_keys,
    selected_work_sources,
)
from ...program.values.transfer import TransferRows, TransferSource
from . import (
    LoadCompletion,
    LoadTimelineProtocol,
    LoadTransfer,
    StoreBatch,
    StoreCompletion,
    StoreTransfer,
)
from .executor import TimelineExecutor

if TYPE_CHECKING:
    from ....attention_fence import AttentionComputeStartGate

_TIMELINE_POLL_INTERVAL_S = 1.0


class LayerwiseBackendOperations(Protocol):
    """Backend session operations required by layerwise timelines.

    Code tuples align exactly with input keys: zero means success, nonzero means
    failure. The Backend boundary rejects malformed results before returning.
    """

    def validate_support(self) -> None: ...

    def start_load_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]: ...

    def finish_load_sessions(self, keys: list[str]) -> None: ...

    def start_store_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]: ...

    def commit_store_sessions(self, keys: list[str]) -> tuple[int, ...]: ...

    def revoke_store_sessions(self, keys: list[str]) -> tuple[int, ...]: ...


class LayerwiseLoadTimelineProtocol(LoadTimelineProtocol, Protocol):
    """Extend Load execution with a visibility fence at each model layer."""

    def wait_for_layer(self, layer_name: str) -> Iterable[LoadCompletion]: ...


class LayerwiseStoreTimelineProtocol(Protocol):
    """Publish Store work through one session and its ordered layer ranges."""

    def bind_admission(self, admission: Callable[[list[StoreTransfer]], list[StoreTransfer]]) -> None: ...

    def bind_operation(self, operation: Callable[[StoreTransfer, Any], StoreCompletion]) -> None: ...

    def start(self) -> None: ...

    def prepare(self, transfers: list[StoreTransfer]) -> None: ...

    def submit_layer(self, layer_name: str, record_source_ready: Callable[[], Any]) -> None: ...

    def finalize(self) -> StoreBatch: ...

    def wait(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]: ...

    def prepare_close(self) -> StoreBatch | None: ...

    def close(self) -> StoreBatch | None: ...


@dataclass(slots=True)
class _OpenLoadSession:
    transfers: tuple[LoadTransfer, ...]
    keys: tuple[str, ...]
    object_sizes: tuple[int, ...]
    completed: threading.Event = field(default_factory=threading.Event)
    session: _LoadSession | None = None
    completions: tuple[LoadCompletion, ...] = ()


@dataclass(slots=True)
class _LoadLayerJob:
    layer_id: int
    transfers: tuple[LoadTransfer, ...]
    session: _LoadSession | None = None
    start_gate: AttentionComputeStartGate | None = None
    completed: threading.Event = field(default_factory=threading.Event)
    completions: list[LoadCompletion] = field(default_factory=list)
    submitted: bool = False
    consumed: bool = False


@dataclass(slots=True)
class _LoadSession:
    """Own one step's Backend read sessions and layer visibility progress."""

    session_keys: tuple[str, ...]
    layer_order: tuple[int, ...]
    layer_jobs: dict[int, _LoadLayerJob]
    next_layer_index: int = 0
    finish_scheduled: bool = False
    finish_completed: threading.Event = field(default_factory=threading.Event)
    session_end_attempted: bool = False
    session_end_confirmed: bool = False
    close_requested: bool = False


@dataclass(frozen=True, slots=True)
class _FinishLoadSession:
    session: _LoadSession


@dataclass(slots=True)
class _StoreSession:
    """Own one step's Store sessions and evidence at each Backend granularity.

    Session results belong to remote objects. Range results belong to one
    remote object and physical layer. Failure and source release remain
    explicit because a Backend may report neither a result code nor an error.
    """

    transfers: tuple[StoreTransfer, ...]
    existing_keys: set[str] = field(default_factory=set)
    pending_finalization_keys: tuple[str, ...] = ()
    pending_transfers_by_layer: dict[int, tuple[StoreTransfer, ...]] = field(default_factory=dict)
    failed_keys: set[str] = field(default_factory=set)
    session_result_codes: dict[str, int | None] = field(default_factory=dict)
    range_result_codes: dict[tuple[str, int], int | None] = field(default_factory=dict)
    transfer_evidence_by_rows: dict[TransferRows, list[TransferEvidence]] = field(default_factory=dict)
    source_release_confirmed_by_source: dict[tuple[int, int, tuple[int, ...], str], bool] = field(default_factory=dict)
    unsafe_rows: set[TransferRows] = field(default_factory=set)
    errors_by_key: dict[str, Exception] = field(default_factory=dict)

    def open(
        self,
        admitted_transfers: tuple[StoreTransfer, ...],
        object_sizes_by_key: dict[str, int],
        backend_io: LayerwiseBackendOperations,
    ) -> None:
        keys = tuple(object_sizes_by_key)
        missing_keys = tuple(dict.fromkeys(_transfer_keys(admitted_transfers)))
        self.existing_keys.update(set(keys) - set(missing_keys))
        if missing_keys:
            self._start_sessions(missing_keys, object_sizes_by_key, backend_io)
        self.pending_transfers_by_layer = _group_layerwise_store_transfers(
            list(admitted_transfers), set(self.pending_finalization_keys)
        )

    def record_range(self, transfer: StoreTransfer, completion: StoreCompletion) -> None:
        transfer_keys = set(_transfer_keys((transfer,)))
        observed_keys: set[str] = set()
        for item in completion.evidence.transfer_evidence:
            self.transfer_evidence_by_rows.setdefault(item.source.rows, []).append(item)
            key = item.source.key
            observed_keys.add(key)
            layer_id = item.source.physical_layer_ids[0]
            self.range_result_codes[key, layer_id] = item.result_code
            if item.result_code != 0:
                self.failed_keys.add(key)
            if item.source_release_confirmed is not True:
                self.source_release_confirmed_by_source[_source_identity(item.source)] = False
        if not observed_keys and not completion.evidence.succeeded:
            self.failed_keys.update(transfer_keys)
        if not observed_keys and not completion.evidence.source_release_confirmed:
            self.unsafe_rows.update(transfer.rows)
        if completion.evidence.error is not None:
            error_keys = observed_keys or transfer_keys
            for key in error_keys:
                self.errors_by_key.setdefault(key, completion.evidence.error)

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
            key for key, result_code in zip(missing_keys, result_codes, strict=True) if result_code == 0
        )
        for key, result_code in zip(missing_keys, result_codes, strict=True):
            if result_code != 0:
                self.failed_keys.add(key)
                self.session_result_codes[key] = result_code

    def _record_session_results(self, keys: list[str], result_codes: tuple[int, ...]) -> None:
        for key, result_code in zip(keys, result_codes, strict=True):
            self.session_result_codes[key] = result_code
            if result_code != 0:
                self.failed_keys.add(key)
        committed_keys = {key for key, result_code in zip(keys, result_codes, strict=True) if result_code == 0}
        self.pending_finalization_keys = tuple(
            key for key in self.pending_finalization_keys if key not in committed_keys
        )

    def _revoke_sessions(self, keys: list[str], backend_io: LayerwiseBackendOperations) -> None:
        try:
            result_codes = backend_io.revoke_store_sessions(keys)
        except Exception as error:
            for key in keys:
                self.errors_by_key.setdefault(key, error)
        else:
            for key, result_code in zip(keys, result_codes, strict=True):
                if result_code != 0:
                    self.session_result_codes.setdefault(key, result_code)
                    self.errors_by_key.setdefault(
                        key,
                        RuntimeError(f"Store session revocation was not confirmed; result code {result_code}"),
                    )
        self.pending_finalization_keys = tuple(key for key in self.pending_finalization_keys if key not in keys)

    def _build_completions(self) -> tuple[StoreCompletion, ...]:
        completions = []
        for transfer in self.transfers:
            recorded = tuple(item for rows in transfer.rows for item in self.transfer_evidence_by_rows.get(rows, ()))
            evidence = tuple(
                TransferEvidence(
                    item.source,
                    self._result_code(item.source),
                    self.source_release_confirmed_by_source.get(
                        _source_identity(item.source),
                        item.source_release_confirmed,
                    ),
                )
                for item in recorded
                if item.source.key not in self.existing_keys
            )
            keys = set(_transfer_keys((transfer,)))
            failed = keys & self.failed_keys
            error = next((self.errors_by_key[key] for key in keys if key in self.errors_by_key), None)
            source_release_confirmed = not any(rows in self.unsafe_rows for rows in transfer.rows) and all(
                item.source_release_confirmed is True for item in evidence
            )
            completions.append(
                StoreCompletion(
                    transfer.request_id,
                    StoreEvidence(evidence, not failed and error is None, source_release_confirmed, error),
                    transfer.store_job_id,
                )
            )
        return tuple(completions)

    def _result_code(self, source: TransferSource) -> int | None:
        key = source.key
        if key in self.session_result_codes:
            return self.session_result_codes[key]
        return self.range_result_codes.get((key, source.physical_layer_ids[0]))


@dataclass(slots=True)
class _OpenStoreSession:
    transfers: tuple[StoreTransfer, ...]
    object_sizes_by_key: dict[str, int]
    completed: threading.Event = field(default_factory=threading.Event)
    session: _StoreSession | None = None


@dataclass(frozen=True, slots=True)
class _StoreLayerJob:
    session: _StoreSession
    transfers: tuple[StoreTransfer, ...]
    source_ready_event: Any


@dataclass(frozen=True, slots=True)
class _FinalizeStoreSession:
    session: _StoreSession
    batch: StoreBatch
    incomplete_layer_ids: tuple[int, ...]


class LayerwiseLoadTimeline:
    """Prefetch layer ranges while preserving vLLM's per-layer visibility fence."""

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
        self._operation: Callable[[LoadTransfer], LoadCompletion] | None = None
        self._session: _LoadSession | None = None
        self._lifecycle_lock = threading.Lock()
        self._executor = TimelineExecutor(
            "KVPoolLayerwiseLoadExecutor", thread_initializer, self._execute, self._complete
        )

    def bind_operation(self, operation: Callable[[LoadTransfer], LoadCompletion]) -> None:
        with self._lifecycle_lock:
            if self._operation is not None:
                raise RuntimeError("Layerwise Load timeline operation is already bound")
            self._operation = operation

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._operation is None:
                raise RuntimeError("Layerwise Load timeline has no bound operation")
            self._backend_io.validate_support()
        self._executor.start()
        self._raise_if_failed()

    def submit(self, transfers: list[LoadTransfer]) -> tuple[LoadCompletion, ...]:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            if self._session is not None:
                raise RuntimeError("Previous Layerwise Load session has not finished")
            work = tuple(item for transfer in transfers for item in transfer.work if not item.empty)
            if not work:
                return tuple(LoadCompletion(transfer.request_id, ()) for transfer in transfers)
            unknown_layer_ids = sorted({item.physical_layer_id for item in work} - set(self._layer_order))
            if unknown_layer_ids:
                raise ValueError(f"Unknown physical Layer IDs {unknown_layer_ids}")
            object_sizes_by_key = _collect_object_sizes(transfers)
            command = _OpenLoadSession(
                tuple(transfers), tuple(object_sizes_by_key), tuple(object_sizes_by_key.values())
            )
            self._executor.submit(command)
            self._wait_for_completion(command.completed)
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
            completions = tuple(layer_job.completions)
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
                self._execute_layer_job(command)
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
        command.completions = tuple(
            LoadCompletion(
                transfer.request_id,
                tuple(
                    TransferEvidence(source, codes_by_key[source.key])
                    for source in selected_work_sources(transfer.work)
                    if codes_by_key[source.key] != 0
                ),
            )
            for transfer in command.transfers
            if any(codes_by_key[key] != 0 for key in _transfer_keys((transfer,)))
        )
        transfers_by_layer = _group_layerwise_transfers(list(command.transfers), set(session_keys))
        layer_order = tuple(layer_id for layer_id in self._layer_order if layer_id in transfers_by_layer)
        if not layer_order:
            return
        layer_jobs = {layer_id: _LoadLayerJob(layer_id, transfers_by_layer[layer_id]) for layer_id in layer_order}
        session = _LoadSession(session_keys, layer_order, layer_jobs)
        for layer_job in layer_jobs.values():
            layer_job.session = session
        command.session = session

    def _execute_layer_job(self, layer_job: _LoadLayerJob) -> None:
        assert self._operation is not None
        session = layer_job.session
        if session is None:
            raise RuntimeError("Layerwise Load range has no owning session")
        if layer_job.start_gate is not None:
            while not layer_job.start_gate.wait(timeout=10):
                logger.info("Layerwise %d load waits for attention compute start", layer_job.layer_id)
        if session.close_requested:
            return
        work = merge_transfer_work(tuple(transfer.work[0] for transfer in layer_job.transfers))
        rows = tuple(rows for transfer in layer_job.transfers for rows in transfer.rows)
        completion = self._operation(LoadTransfer("<layer-batch>", rows, (work,)))
        for transfer in layer_job.transfers:
            transfer_rows = {id(rows) for rows in transfer.rows}
            evidence = tuple(item for item in completion.transfer_evidence if id(item.source.rows) in transfer_rows)
            layer_job.completions.append(LoadCompletion(transfer.request_id, evidence))

    def _fill_prefetch_window(
        self,
        session: _LoadSession,
        current_layer_id: int,
        start_gate: AttentionComputeStartGate,
    ) -> None:
        active_ranges = sum(item.submitted and not item.consumed for item in session.layer_jobs.values())
        while active_ranges < self._prefetch_layers and session.next_layer_index < len(session.layer_order):
            layer_id = session.layer_order[session.next_layer_index]
            session.next_layer_index += 1
            layer_job = session.layer_jobs[layer_id]
            if not layer_job.submitted:
                layer_job.start_gate = None if layer_id == current_layer_id else start_gate
                layer_job.submitted = True
                self._executor.submit(layer_job)
                active_ranges += 1
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
            layer_job = session.layer_jobs[next_layer_id]
            if not layer_job.submitted:
                layer_job.start_gate = None if next_layer_id == layer_id else start_gate
                layer_job.submitted = True
                self._executor.submit(layer_job)
        self._schedule_finish_if_complete(session)

    def _request_finish(self, session: _LoadSession) -> None:
        session.close_requested = True
        for layer_job in session.layer_jobs.values():
            if layer_job.submitted and not layer_job.completed.is_set() and layer_job.start_gate is not None:
                layer_job.start_gate.cancel()
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
        # Retain layer jobs until the executor publishes failure; None would let concurrent fences skip waiting.
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


class LayerwiseStoreTimeline:
    """Publish layer-restricted Store ranges and commit their shared objects."""

    def __init__(
        self,
        topology: KVPoolTopology,
        backend_io: LayerwiseBackendOperations,
        thread_initializer: Callable[[], None],
    ) -> None:
        self._backend_io = backend_io
        self._layer_ids_by_name = _compile_layer_ids_by_name(topology)
        self._admission: Callable[[list[StoreTransfer]], list[StoreTransfer]] | None = None
        self._operation: Callable[[StoreTransfer, Any], StoreCompletion] | None = None
        self._lifecycle_lock = threading.Lock()
        self._session: _StoreSession | None = None
        self._pending_batch: StoreBatch | None = None
        self._executor = TimelineExecutor(
            "KVPoolLayerwiseStoreExecutor", thread_initializer, self._execute, self._complete
        )

    def bind_admission(self, admission: Callable[[list[StoreTransfer]], list[StoreTransfer]]) -> None:
        with self._lifecycle_lock:
            if self._admission is not None:
                raise RuntimeError("Layerwise Store timeline admission is already bound")
            self._admission = admission

    def bind_operation(self, operation: Callable[[StoreTransfer, Any], StoreCompletion]) -> None:
        with self._lifecycle_lock:
            if self._operation is not None:
                raise RuntimeError("Layerwise Store timeline operation is already bound")
            self._operation = operation

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._admission is None:
                raise RuntimeError("Layerwise Store timeline has no bound admission")
            if self._operation is None:
                raise RuntimeError("Layerwise Store timeline has no bound operation")
            self._backend_io.validate_support()
        self._executor.start()
        self._raise_if_failed()

    def prepare(self, transfers: list[StoreTransfer]) -> None:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            if self._session is not None or self._pending_batch is not None:
                raise RuntimeError("Previous Layerwise Store session has not reached its fence")
            self._session = self._open_session(transfers)

    def submit_layer(self, layer_name: str, record_source_ready: Callable[[], Any]) -> None:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            session = self._session
            if session is None:
                return
            layer_id = self._layer_ids_by_name.get(layer_name)
            if layer_id is None:
                return
            layer_transfers = session.pending_transfers_by_layer.get(layer_id, ())
            if layer_transfers:
                # Record on the caller's stream before consuming work; event failures must leave it pending.
                source_ready_event = record_source_ready()
                session.pending_transfers_by_layer.pop(layer_id)
                self._executor.submit(_StoreLayerJob(session, layer_transfers, source_ready_event))

    def finalize(self) -> StoreBatch:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            return self._finalize_session()

    def _finalize_session(self) -> StoreBatch:
        session = self._session
        if session is None:
            raise RuntimeError("Layerwise Store session has not been prepared")
        batch = StoreBatch(session.transfers)
        self._session = None
        self._pending_batch = batch
        incomplete_layer_ids = tuple(sorted(session.pending_transfers_by_layer))
        self._executor.submit(_FinalizeStoreSession(session, batch, incomplete_layer_ids))
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
        """Expose the finalization fence, including sessions interrupted before finalize."""

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
                session = _StoreSession(command.transfers)
                command.session = session
                admitted_transfers = tuple(self._admission(list(command.transfers)))
                session.open(admitted_transfers, command.object_sizes_by_key, self._backend_io)
            elif isinstance(command, _StoreLayerJob):
                self._execute_layer_job(command)
            else:
                command.batch.completions.extend(
                    command.session.finalize(self._backend_io, command.incomplete_layer_ids)
                )
        except BaseException:
            failed_session = command.session
            if failed_session is not None:
                failed_session.revoke_pending(self._backend_io)
            raise

    @staticmethod
    def _complete(command: _OpenStoreSession | _StoreLayerJob | _FinalizeStoreSession) -> None:
        if isinstance(command, _OpenStoreSession):
            command.completed.set()
        elif isinstance(command, _FinalizeStoreSession):
            command.batch.completed.set()

    def _open_session(self, transfers: list[StoreTransfer]) -> _StoreSession:
        command = _OpenStoreSession(tuple(transfers), _collect_object_sizes(transfers))
        self._executor.submit(command)
        self._wait_for_completion(command.completed)
        if command.session is None:
            raise RuntimeError("Layerwise Store executor did not create a session")
        return command.session

    def _execute_layer_job(self, job: _StoreLayerJob) -> None:
        assert self._operation is not None
        work = merge_transfer_work(tuple(transfer.work[0] for transfer in job.transfers))
        rows = tuple(rows for transfer in job.transfers for rows in transfer.rows)
        aggregate = StoreTransfer("<layer-batch>", rows, (work,))
        completion = self._operation(aggregate, job.source_ready_event)
        for transfer in job.transfers:
            transfer_rows = {id(rows) for rows in transfer.rows}
            evidence = tuple(
                item for item in completion.evidence.transfer_evidence if id(item.source.rows) in transfer_rows
            )
            error = completion.evidence.error
            succeeded = error is None and bool(evidence) and all(item.result_code == 0 for item in evidence)
            source_release_confirmed = (
                all(item.source_release_confirmed is True for item in evidence)
                if evidence
                else completion.evidence.source_release_confirmed
            )
            split = StoreCompletion(
                transfer.request_id,
                StoreEvidence(
                    evidence,
                    succeeded,
                    source_release_confirmed,
                    error,
                ),
                transfer.store_job_id,
            )
            job.session.record_range(transfer, split)

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


def _collect_object_sizes(transfers: Iterable[LoadTransfer | StoreTransfer]) -> dict[str, int]:
    object_sizes_by_key: dict[str, int] = {}
    for transfer in transfers:
        for rows in transfer.rows:
            for coordinate_index, keys in enumerate(rows.keys_by_coordinate):
                object_size = rows.plan.object_sizes[coordinate_index]
                for key in keys:
                    previous_size = object_sizes_by_key.setdefault(key, object_size)
                    if previous_size != object_size:
                        raise ValueError(f"Layerwise key {key!r} has inconsistent object sizes")
    return object_sizes_by_key


def _transfer_keys(transfers: Iterable[LoadTransfer | StoreTransfer]) -> tuple[str, ...]:
    return tuple(key for transfer in transfers for key in selected_work_keys(transfer.work))


def _source_identity(source: TransferSource) -> tuple[int, int, tuple[int, ...], str]:
    """Identify one local source independently from its shared remote key."""

    return id(source.rows), source.row_index, source.physical_layer_ids, source.key


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


def _group_layerwise_transfers(
    transfers: list[LoadTransfer], session_keys: set[str]
) -> dict[int, tuple[LoadTransfer, ...]]:
    grouped: dict[int, list[LoadTransfer]] = {}
    for transfer in transfers:
        selected_work = select_work_keys(transfer.work, session_keys)
        for work in selected_work:
            if work.physical_layer_id is None:
                raise ValueError("Layerwise Load received bulk execution work")
            if not work.empty:
                grouped.setdefault(work.physical_layer_id, []).append(
                    LoadTransfer(transfer.request_id, transfer.rows, (work,))
                )
    return {layer_id: tuple(layer_transfers) for layer_id, layer_transfers in grouped.items()}


def _group_layerwise_store_transfers(
    transfers: list[StoreTransfer], session_keys: set[str]
) -> dict[int, tuple[StoreTransfer, ...]]:
    grouped: dict[int, list[StoreTransfer]] = {}
    claimed_keys: set[str] = set()
    for transfer in transfers:
        selected_work = select_work_keys(transfer.work, session_keys, claimed_keys)
        for work in selected_work:
            if work.physical_layer_id is None:
                raise ValueError("Layerwise Store received bulk execution work")
            if not work.empty:
                grouped.setdefault(work.physical_layer_id, []).append(
                    StoreTransfer(transfer.request_id, transfer.rows, (work,), transfer.store_job_id)
                )
    return {layer_id: tuple(layer_transfers) for layer_id, layer_transfers in grouped.items()}
