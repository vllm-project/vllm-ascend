"""Coordinate layer-triggered Store with session-wide visibility fences."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from vllm.logger import logger

from ..protocol.transfer import StoreCommand
from ..topology import KVPoolTopology
from ..worker.transfer.batch import KVTransferBatch, TransferSource
from ..worker.transfer.evidence import LayerStoreResult, StoreCompletion, StoreEvidence, TransferEvidence
from .executor import TimelineExecutor
from .layerwise_common import collect_object_sizes, compile_layer_ids_by_name
from .store_batch import StoreBatch

_TIMELINE_POLL_INTERVAL_S = 1.0

StorePreparation = Callable[[tuple[StoreCommand, ...]], tuple[KVTransferBatch | None, tuple[StoreCompletion, ...]]]
StoreLayerOperation = Callable[[Any, int], LayerStoreResult]
StartStoreSessions = Callable[[list[str], list[int]], tuple[int, ...]]
PrepareStoreLayers = Callable[[KVTransferBatch], dict[int, list[str]]]
FinishStoreSessions = Callable[[list[str]], tuple[int, ...]]


@dataclass(slots=True)
class _StoreSession:
    commands: tuple[StoreCommand, ...]
    batch: KVTransferBatch | None = None
    terminal_completions: tuple[StoreCompletion, ...] = ()
    pending_finalization_keys: tuple[str, ...] = ()
    selected: KVTransferBatch | None = None
    request_keys: tuple[frozenset[str], ...] = ()
    sources_by_request: tuple[tuple[TransferSource, ...], ...] = ()
    expected_keys_by_layer: dict[int, list[str]] = field(default_factory=dict)
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
        start_sessions: StartStoreSessions,
        prepare_layers: PrepareStoreLayers,
        revoke_sessions: FinishStoreSessions,
    ) -> None:
        self.batch = batch
        self.terminal_completions = terminal_completions
        if batch is None:
            return
        object_sizes_by_key = collect_object_sizes(batch)
        selected_keys = batch.selected_keys()
        session_keys = tuple(dict.fromkeys(selected_keys))
        if session_keys:
            self._start_sessions(session_keys, object_sizes_by_key, start_sessions, revoke_sessions)
        self.selected = (
            batch
            if self.pending_finalization_keys == selected_keys
            else batch.select_keys(set(self.pending_finalization_keys), claim_once=True)
        )
        if self.selected is batch:
            self.request_keys, self.sources_by_request = _request_state(batch)
        else:
            self.request_keys = _keys_by_request(batch)
            self.sources_by_request = _sources_by_request(self.selected)
        try:
            self.expected_keys_by_layer = prepare_layers(self.selected)
        except Exception as error:
            self.failed_keys.update(self.pending_finalization_keys)
            self.errors_by_key.update((key, error) for key in self.pending_finalization_keys)
            return
        self.pending_layer_ids = set(self.expected_keys_by_layer)

    def record_range(self, result: LayerStoreResult, layer_id: int, expected_keys: Sequence[str]) -> None:
        if self.batch is None:
            raise RuntimeError("Layerwise Store range completed without a prepared batch")
        self.observed_layer_ids.add(layer_id)
        if isinstance(result.result_codes, tuple):
            if not isinstance(result.source_release_confirmed, tuple) or not (
                len(expected_keys) == len(result.result_codes) == len(result.source_release_confirmed)
            ):
                raise RuntimeError("Layerwise Store result axes are not aligned")
            results = zip(expected_keys, result.result_codes, result.source_release_confirmed, strict=True)
        else:
            if not isinstance(result.source_release_confirmed, bool):
                raise RuntimeError("Layerwise Store scalar result has non-scalar source evidence")
            code = result.result_codes
            released = result.source_release_confirmed
            if code != 0:
                self.failed_keys.update(expected_keys)
                for key in expected_keys:
                    self.range_failure_codes.setdefault(key, code)
            if not released:
                self.unsafe_keys.update(expected_keys)
            if result.error is not None:
                for key in expected_keys:
                    self.errors_by_key.setdefault(key, result.error)
            return
        for key, code, released in results:
            if code != 0:
                self.failed_keys.add(key)
                self.range_failure_codes.setdefault(key, code)
            if not released:
                self.unsafe_keys.add(key)
            if result.error is not None:
                self.errors_by_key.setdefault(key, result.error)

    def finalize(
        self,
        commit_sessions: FinishStoreSessions,
        revoke_sessions: FinishStoreSessions,
    ) -> tuple[StoreCompletion, ...]:
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
                result_codes = commit_sessions(commit_keys)
            except Exception as error:
                self.failed_keys.update(commit_keys)
                self.errors_by_key.update((key, error) for key in commit_keys)
            else:
                self._record_session_results(commit_keys, result_codes)
        revoke_keys = [key for key in self.pending_finalization_keys if key in self.failed_keys]
        if revoke_keys:
            self._revoke_sessions(revoke_keys, revoke_sessions)
        return self._build_completions()

    def revoke_pending(self, revoke_sessions: FinishStoreSessions) -> None:
        if not self.pending_finalization_keys:
            return
        try:
            revoke_sessions(list(self.pending_finalization_keys))
        except BaseException:
            logger.exception("Failed to revoke Layerwise Store sessions after a timeline error")

    def _start_sessions(
        self,
        missing_keys: tuple[str, ...],
        object_sizes_by_key: dict[str, int],
        start_sessions: StartStoreSessions,
        revoke_sessions: FinishStoreSessions,
    ) -> None:
        try:
            result_codes = start_sessions(list(missing_keys), [object_sizes_by_key[key] for key in missing_keys])
        except Exception as error:
            self.failed_keys.update(missing_keys)
            self.errors_by_key.update((key, error) for key in missing_keys)
            self._revoke_sessions(list(missing_keys), revoke_sessions)
            return
        self.pending_finalization_keys = tuple(
            key for key, code in zip(missing_keys, result_codes, strict=True) if code == 0
        )
        for key, code in zip(missing_keys, result_codes, strict=True):
            if code != 0:
                self.failed_keys.add(key)
                self.session_result_codes[key] = code
                self.errors_by_key[key] = RuntimeError(
                    f"Store session start failed for key {key!r}; result code {code}"
                )

    def _record_session_results(self, keys: list[str], result_codes: tuple[int, ...]) -> None:
        for key, code in zip(keys, result_codes, strict=True):
            self.session_result_codes[key] = code
            if code != 0:
                self.failed_keys.add(key)
        committed = {key for key, code in zip(keys, result_codes, strict=True) if code == 0}
        self.pending_finalization_keys = tuple(key for key in self.pending_finalization_keys if key not in committed)

    def _revoke_sessions(self, keys: list[str], revoke_sessions: FinishStoreSessions) -> None:
        try:
            result_codes = revoke_sessions(keys)
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
        sources_by_request = _sources_for_observed_layers(self.sources_by_request, self.observed_layer_ids)
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
        self,
        topology: KVPoolTopology,
        thread_initializer: Callable[[], None],
        preparation: StorePreparation,
        operation: StoreLayerOperation,
        start_sessions: StartStoreSessions,
        prepare_layers: PrepareStoreLayers,
        commit_sessions: FinishStoreSessions,
        revoke_sessions: FinishStoreSessions,
    ) -> None:
        self._preparation = preparation
        self._operation = operation
        self._start_sessions = start_sessions
        self._prepare_layers = prepare_layers
        self._commit_sessions = commit_sessions
        self._revoke_sessions = revoke_sessions
        self._layer_ids_by_name = compile_layer_ids_by_name(topology)
        self._lifecycle_lock = threading.Lock()
        self._session: _StoreSession | None = None
        self._pending_batch: StoreBatch | None = None
        self._executor = TimelineExecutor(
            "KVPoolLayerwiseStoreExecutor", thread_initializer, self._execute, self._complete
        )

    def start(self) -> None:
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

    @property
    def stopped(self) -> bool:
        return self._executor.stopped

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
                batch, completions = self._preparation(command.session.commands)
                command.session.open(
                    batch,
                    completions,
                    self._start_sessions,
                    self._prepare_layers,
                    self._revoke_sessions,
                )
            elif isinstance(command, _StoreLayerJob):
                self._execute_layer(command)
            else:
                command.batch.completions.extend(command.session.finalize(self._commit_sessions, self._revoke_sessions))
        except BaseException:
            command.session.revoke_pending(self._revoke_sessions)
            raise

    @staticmethod
    def _complete(command: _OpenStoreSession | _StoreLayerJob | _FinalizeStoreSession) -> None:
        if isinstance(command, _FinalizeStoreSession):
            command.batch.completed.set()

    def _execute_layer(self, job: _StoreLayerJob) -> None:
        if job.layer_id not in job.session.pending_layer_ids:
            return
        job.session.pending_layer_ids.remove(job.layer_id)
        expected_keys = job.session.expected_keys_by_layer[job.layer_id]
        result = self._operation(job.source_ready_event, job.layer_id)
        job.session.record_range(result, job.layer_id, expected_keys)

    def _raise_if_failed(self) -> None:
        if self._executor.failure is not None:
            raise RuntimeError("Layerwise Store failed during asynchronous Store") from self._executor.failure

    def _raise_if_not_running(self) -> None:
        self._raise_if_failed()
        self._executor.check_running()


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
                    if group.selected_objects is None or group.selected_objects[object_index]:
                        keys.add(axis[row_index])
    return tuple(frozenset(keys) for keys in keys_by_request)


def _request_state(
    batch: KVTransferBatch,
) -> tuple[tuple[frozenset[str], ...], tuple[tuple[TransferSource, ...], ...]]:
    keys_by_request: list[set[str]] = [set() for _ in batch.request_ids]
    sources_by_request: list[list[TransferSource]] = [[] for _ in batch.request_ids]
    for group in batch.groups:
        for request_index, (keys, sources) in enumerate(zip(keys_by_request, sources_by_request, strict=True)):
            start = int(group.request_splits[request_index])
            end = int(group.request_splits[request_index + 1])
            for axis_index, axis in enumerate(group.key_axes):
                axis_offset = axis_index * group.row_count
                for row_index in range(start, end):
                    object_index = axis_offset + row_index
                    if group.selected_objects is not None and not group.selected_objects[object_index]:
                        continue
                    key = axis[row_index]
                    keys.add(key)
                    sources.append(
                        TransferSource(
                            group,
                            object_index,
                            row_index,
                            request_index,
                            group.physical_layer_ids,
                            key,
                        )
                    )
    return (
        tuple(frozenset(keys) for keys in keys_by_request),
        tuple(tuple(sources) for sources in sources_by_request),
    )


def _sources_by_request(batch: KVTransferBatch) -> tuple[tuple[TransferSource, ...], ...]:
    sources_by_request: list[list[TransferSource]] = [[] for _ in batch.request_ids]
    for group in batch.groups:
        for request_index, sources in enumerate(sources_by_request):
            start = int(group.request_splits[request_index])
            end = int(group.request_splits[request_index + 1])
            for axis_index, axis in enumerate(group.key_axes):
                axis_offset = axis_index * group.row_count
                for row_index in range(start, end):
                    object_index = axis_offset + row_index
                    if group.selected_objects is None or group.selected_objects[object_index]:
                        source = TransferSource(
                            group, object_index, row_index, request_index, group.physical_layer_ids, axis[row_index]
                        )
                        sources.append(source)
    return tuple(tuple(sources) for sources in sources_by_request)


def _sources_for_observed_layers(
    sources_by_request: tuple[tuple[TransferSource, ...], ...], observed_layer_ids: set[int]
) -> tuple[tuple[TransferSource, ...], ...]:
    observed_by_request = []
    for sources in sources_by_request:
        observed_sources = []
        for source in sources:
            physical_layer_ids = tuple(
                layer_id for layer_id in source.physical_layer_ids if layer_id in observed_layer_ids
            )
            if not physical_layer_ids:
                continue
            observed_sources.append(
                source
                if physical_layer_ids == source.physical_layer_ids
                else replace(source, physical_layer_ids=physical_layer_ids)
            )
        observed_by_request.append(tuple(observed_sources))
    return tuple(observed_by_request)
