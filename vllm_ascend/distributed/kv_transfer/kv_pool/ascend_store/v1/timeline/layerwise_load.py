"""Coordinate layer-triggered Load with session-wide visibility fences."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from vllm.logger import logger

from ..topology import KVPoolTopology
from ..worker.transfer.batch import KVTransferBatch
from ..worker.transfer.evidence import LoadCompletion, TransferEvidence
from .executor import TimelineExecutor
from .layerwise_common import collect_object_sizes, compile_layer_ids_by_name

if TYPE_CHECKING:
    from ...attention_fence import AttentionComputeStartGate

_TIMELINE_POLL_INTERVAL_S = 1.0


LoadOperation = Callable[[int], tuple[LoadCompletion, ...]]
StartLoadSessions = Callable[[list[str], list[int]], tuple[int, ...]]
PrepareLoadLayers = Callable[[KVTransferBatch], dict[int, list[str]]]
FinishLoadSessions = Callable[[list[str]], None]


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
        prefetch_layers: int,
        thread_initializer: Callable[[], None],
        start_gate_factory: Callable[[], AttentionComputeStartGate],
        operation: LoadOperation,
        start_sessions: StartLoadSessions,
        prepare_layers: PrepareLoadLayers,
        finish_sessions: FinishLoadSessions,
    ) -> None:
        if prefetch_layers <= 0:
            raise ValueError("Layerwise Load prefetch depth must be positive")
        self._operation = operation
        self._start_sessions = start_sessions
        self._prepare_layers = prepare_layers
        self._finish_sessions = finish_sessions
        self._layer_ids_by_name = compile_layer_ids_by_name(topology)
        self._layer_order = tuple(sorted(set(self._layer_ids_by_name.values())))
        self._prefetch_layers = prefetch_layers
        self._start_gate_factory = start_gate_factory
        self._session: _LoadSession | None = None
        self._lifecycle_lock = threading.Lock()
        self._executor = TimelineExecutor(
            "KVPoolLayerwiseLoadExecutor", thread_initializer, self._execute, self._complete
        )

    def start(self) -> None:
        self._executor.start()
        self._raise_if_failed()

    def submit(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        with self._lifecycle_lock:
            self._raise_if_not_running()
            if self._session is not None:
                raise RuntimeError("Previous Layerwise Load session has not completed")
            object_sizes = collect_object_sizes(batch)
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

    @property
    def stopped(self) -> bool:
        return self._executor.stopped

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._executor.stopped:
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
        result_codes = self._start_sessions(list(command.keys), list(command.object_sizes))
        codes_by_key = dict(zip(command.keys, result_codes, strict=True))
        session_keys = tuple(key for key, code in codes_by_key.items() if code == 0)
        command.completions = _load_session_failures(command.batch, codes_by_key)
        selected = command.batch.select_keys(set(session_keys))
        keys_by_layer = self._prepare_layers(selected)
        layer_order = tuple(layer_id for layer_id in self._layer_order if keys_by_layer.get(layer_id))
        if not layer_order:
            self._finish_sessions(list(session_keys))
            return
        session = _LoadSession(selected, session_keys, layer_order)
        session.layer_jobs = {layer_id: _LoadLayerJob(layer_id, session) for layer_id in layer_order}
        command.session = session

    def _execute_layer(self, job: _LoadLayerJob) -> None:
        if job.start_gate is not None:
            while not job.start_gate.wait(timeout=10):
                logger.info("Layerwise %d load waits for attention compute start", job.layer_id)
        if job.session.close_requested:
            return
        job.completions = self._operation(job.layer_id)

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
        self._finish_sessions(list(session.session_keys))
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
                self._finish_sessions(list(cleanup_keys))
                if session is not None:
                    session.session_end_confirmed = True
            except BaseException:
                logger.exception("Failed to close Layerwise Load sessions after an execution error")


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
