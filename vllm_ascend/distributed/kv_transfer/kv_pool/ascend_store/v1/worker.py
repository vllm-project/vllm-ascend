"""Worker-side components for the AscendStore v1 classic path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Condition, Lock
from typing import Protocol

from .execution import LoadExecutor, LookupExecutor, OperationResolver, StoreExecutor
from .protocol import (
    ExecutionOutcome,
    ExecutionStatus,
    LoadRequest,
    LoadResult,
    LoadTask,
    LookupExecutionSucceeded,
    LookupRequest,
    LookupResponse,
    LookupResult,
    LookupTask,
    LookupUnavailable,
    OperationRef,
    SourceReady,
    SourceReleased,
    SourceReleaseUnknown,
    StoreAccepted,
    StoreRequest,
    StoreResult,
    StoreTask,
)


class _OperationKind(str, Enum):
    LOOKUP = "lookup"
    LOAD = "load"
    STORE = "store"


class _OperationPhase(str, Enum):
    EXECUTING = "executing"
    ACCEPTED = "accepted"
    SOURCE_RELEASED = "source_released"
    SOURCE_RELEASE_UNKNOWN = "source_release_unknown"
    COMPLETED = "completed"


class _WorkerLifecycleState(str, Enum):
    RUNNING = "running"
    DRAINING = "draining"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(slots=True)
class _OperationRecord:
    """Hold WorkerService-owned state for one in-flight operation."""

    operation_ref: OperationRef
    kind: _OperationKind
    task: LookupTask | LoadTask | StoreTask
    phase: _OperationPhase
    source_release: SourceReleased | SourceReleaseUnknown | None = None
    result: LookupResponse | LoadResult | StoreResult | None = None


class StoreCompletionPublisher(Protocol):
    """Publish asynchronous Store completion facts to the worker adapter boundary."""

    def publish_source_released(self, source_released: SourceReleased) -> None: ...

    def publish_source_release_unknown(self, source_release_unknown: SourceReleaseUnknown) -> None: ...

    def publish_store_result(self, result: StoreResult) -> None: ...


class WorkerAdapter:
    """Boundary component for model runner hooks."""

    def __init__(self, service: WorkerService) -> None:
        self._service = service


class WorkerService:
    """Worker-side AscendStore service boundary."""

    def __init__(
        self,
        operation_resolver: OperationResolver,
        lookup_executor: LookupExecutor,
        load_executor: LoadExecutor,
        store_executor: StoreExecutor,
        store_completion_publisher: StoreCompletionPublisher,
    ) -> None:
        self._operation_resolver = operation_resolver
        self._lookup_executor = lookup_executor
        self._load_executor = load_executor
        self._store_executor = store_executor
        self._store_completion_publisher = store_completion_publisher
        self._operation_records: dict[str, _OperationRecord] = {}
        self._operation_lock = Lock()
        self._operation_state_changed = Condition(self._operation_lock)
        self._lifecycle_state = _WorkerLifecycleState.RUNNING
        self._lifecycle_failure: str | None = None

    def lookup(self, request: LookupRequest) -> LookupResponse:
        task = self._operation_resolver.resolve_lookup(request)
        self._register_operation(_OperationKind.LOOKUP, task, _OperationPhase.EXECUTING)
        outcome = self._lookup_executor.execute(task)

        if isinstance(outcome, LookupExecutionSucceeded):
            observed_prefix_end = request.candidate_span.start
            for candidate, is_hit in zip(task.candidates, outcome.candidate_hits, strict=True):
                if not is_hit:
                    break
                observed_prefix_end = candidate.token_span.end

            result: LookupResponse = LookupResult(
                operation_ref=request.operation_ref, observed_prefix_end=observed_prefix_end
            )
        else:
            result = LookupUnavailable(
                operation_ref=request.operation_ref, status=outcome.status, detail=outcome.detail
            )

        self._complete_operation(result)
        self._retire_operation(request.operation_ref)
        return result

    def load(self, request: LoadRequest) -> LoadResult:
        task = self._operation_resolver.resolve_load(request)
        self._register_operation(_OperationKind.LOAD, task, _OperationPhase.EXECUTING)
        outcome = self._load_executor.execute(task)

        ready_prefix_end = request.load_span.start
        prefix_is_contiguous = True
        invalid_destination_blocks = []

        for item, item_outcome in zip(task.destination_items, outcome.items, strict=True):
            item_succeeded = item_outcome.status is ExecutionStatus.SUCCEEDED
            if prefix_is_contiguous and item_succeeded:
                ready_prefix_end = item.token_span.end
            else:
                prefix_is_contiguous = False

            if not item_succeeded:
                invalid_destination_blocks.append(item.local_block)

        result = LoadResult(
            operation_ref=request.operation_ref,
            ready_prefix_end=ready_prefix_end,
            invalid_destination_blocks=tuple(invalid_destination_blocks),
            outcome=outcome,
        )
        self._complete_operation(result)
        self._retire_operation(request.operation_ref)
        return result

    def store(self, request: StoreRequest, source_ready: SourceReady) -> StoreAccepted:
        task = self._operation_resolver.resolve_store(request, source_ready)
        self._register_operation(_OperationKind.STORE, task, _OperationPhase.ACCEPTED)
        self._store_executor.submit(
            task,
            self._publish_source_released,
            self._publish_source_release_unknown,
            self._publish_store_result,
        )
        return StoreAccepted(operation_ref=request.operation_ref)

    def close(self) -> None:
        """Drain accepted operations and close execution unless source safety is unknown."""

        with self._operation_state_changed:
            if self._lifecycle_state is _WorkerLifecycleState.CLOSED:
                return
            self._raise_if_worker_failed()
            if self._lifecycle_state is _WorkerLifecycleState.DRAINING:
                self._operation_state_changed.wait_for(
                    lambda: self._lifecycle_state in (_WorkerLifecycleState.CLOSED, _WorkerLifecycleState.FAILED)
                )
                self._raise_if_worker_failed()
                return

            self._lifecycle_state = _WorkerLifecycleState.DRAINING
            self._operation_state_changed.wait_for(
                lambda: not self._operation_records or self._lifecycle_state is _WorkerLifecycleState.FAILED
            )
            self._raise_if_worker_failed()

        self._store_executor.close()

        with self._operation_state_changed:
            self._lifecycle_state = _WorkerLifecycleState.CLOSED
            self._operation_state_changed.notify_all()

    def _raise_if_worker_failed(self) -> None:
        if self._lifecycle_state is _WorkerLifecycleState.FAILED:
            raise RuntimeError(self._lifecycle_failure or "worker service failed")

    def _publish_source_released(self, operation_ref: OperationRef) -> None:
        source_released = SourceReleased(operation_ref=operation_ref)
        self._record_source_released(source_released)
        self._store_completion_publisher.publish_source_released(source_released)

    def _publish_source_release_unknown(self, operation_ref: OperationRef, detail: str) -> None:
        source_release_unknown = SourceReleaseUnknown(operation_ref=operation_ref, detail=detail)
        self._record_source_release_unknown(source_release_unknown)
        self._store_completion_publisher.publish_source_release_unknown(source_release_unknown)

    def _publish_store_result(self, operation_ref: OperationRef, outcome: ExecutionOutcome) -> None:
        result = StoreResult(operation_ref=operation_ref, outcome=outcome)
        self._complete_operation(result)
        self._store_completion_publisher.publish_store_result(result)
        if self._store_source_is_released(operation_ref):
            self._retire_operation(operation_ref)

    def _register_operation(
        self,
        kind: _OperationKind,
        task: LookupTask | LoadTask | StoreTask,
        phase: _OperationPhase,
    ) -> None:
        operation_id = task.operation_ref.operation_id
        record = _OperationRecord(operation_ref=task.operation_ref, kind=kind, task=task, phase=phase)

        with self._operation_state_changed:
            if self._lifecycle_state is not _WorkerLifecycleState.RUNNING:
                state = self._lifecycle_state.value
                raise RuntimeError(f"worker service cannot accept operation {operation_id} while {state}")
            if operation_id in self._operation_records:
                raise ValueError(f"operation {operation_id} is already in flight")
            self._operation_records[operation_id] = record

    def _record_source_released(self, source_released: SourceReleased) -> None:
        with self._operation_state_changed:
            record = self._require_operation_record(source_released.operation_ref)
            if record.kind is not _OperationKind.STORE:
                raise RuntimeError(f"{record.kind.value} operation cannot release source memory")
            self._require_operation_phase(record, _OperationPhase.ACCEPTED)
            record.source_release = source_released
            record.phase = _OperationPhase.SOURCE_RELEASED

    def _record_source_release_unknown(self, source_release_unknown: SourceReleaseUnknown) -> None:
        with self._operation_state_changed:
            record = self._require_operation_record(source_release_unknown.operation_ref)
            if record.kind is not _OperationKind.STORE:
                raise RuntimeError(f"{record.kind.value} operation cannot have unknown source release")
            self._require_operation_phase(record, _OperationPhase.ACCEPTED)
            record.source_release = source_release_unknown
            record.phase = _OperationPhase.SOURCE_RELEASE_UNKNOWN
            self._lifecycle_state = _WorkerLifecycleState.FAILED
            if self._lifecycle_failure is None:
                operation_id = source_release_unknown.operation_ref.operation_id
                self._lifecycle_failure = (
                    f"store operation {operation_id} has unknown source release: {source_release_unknown.detail}"
                )
            self._operation_state_changed.notify_all()

    def _complete_operation(self, result: LookupResponse | LoadResult | StoreResult) -> None:
        with self._operation_state_changed:
            record = self._require_operation_record(result.operation_ref)
            expected_phases = (
                (_OperationPhase.SOURCE_RELEASED, _OperationPhase.SOURCE_RELEASE_UNKNOWN)
                if record.kind is _OperationKind.STORE
                else (_OperationPhase.EXECUTING,)
            )
            self._require_operation_phase(record, *expected_phases)
            record.result = result
            record.phase = _OperationPhase.COMPLETED

    def _retire_operation(self, operation_ref: OperationRef) -> None:
        with self._operation_state_changed:
            record = self._require_operation_record(operation_ref)
            self._require_operation_phase(record, _OperationPhase.COMPLETED)
            if record.result is None:
                raise RuntimeError(f"operation {operation_ref.operation_id} has no result")
            if record.kind is _OperationKind.STORE and not isinstance(record.source_release, SourceReleased):
                raise RuntimeError(f"store operation {operation_ref.operation_id} has not released its source")
            del self._operation_records[operation_ref.operation_id]
            if not self._operation_records:
                self._operation_state_changed.notify_all()

    def _store_source_is_released(self, operation_ref: OperationRef) -> bool:
        with self._operation_state_changed:
            record = self._require_operation_record(operation_ref)
            return isinstance(record.source_release, SourceReleased)

    def _require_operation_record(self, operation_ref: OperationRef) -> _OperationRecord:
        record = self._operation_records.get(operation_ref.operation_id)
        if record is None:
            raise RuntimeError(f"operation {operation_ref.operation_id} is not in flight")
        if record.operation_ref != operation_ref:
            raise RuntimeError(
                f"operation {operation_ref.operation_id} belongs to request {record.operation_ref.request_id}, "
                f"not {operation_ref.request_id}"
            )
        return record

    @staticmethod
    def _require_operation_phase(record: _OperationRecord, *expected_phases: _OperationPhase) -> None:
        if record.phase not in expected_phases:
            operation_id = record.operation_ref.operation_id
            expected = " or ".join(phase.value for phase in expected_phases)
            raise RuntimeError(f"operation {operation_id} is {record.phase.value}; expected {expected}")
