"""Worker-local execution components for the AscendStore v1 classic path."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from .backend import KVStoreBackend
from .protocol import (
    ContentId,
    ExecutionItemOutcome,
    ExecutionOutcome,
    ExecutionStatus,
    LoadRequest,
    LoadTask,
    LocalBlockRef,
    LookupCandidate,
    LookupExecutionOutcome,
    LookupExecutionSucceeded,
    LookupExecutionUnavailable,
    LookupRequest,
    LookupTask,
    MemorySegment,
    OperationRef,
    SourceReady,
    StoreRequest,
    StoreTask,
    TokenSpan,
    TransferItem,
)

STORE_BACKGROUND_WORKER_COUNT = 1


def _aligned_transfer_outcome_or_unknown(
    items: tuple[TransferItem, ...], outcome: ExecutionOutcome
) -> ExecutionOutcome:
    expected_keys = tuple(item.backend_key for item in items)
    returned_keys = tuple(item.backend_key for item in outcome.items)
    if returned_keys == expected_keys:
        return outcome

    detail = "backend outcomes do not match the requested key order"
    unknown_items = tuple(
        ExecutionItemOutcome(backend_key=item.backend_key, status=ExecutionStatus.UNKNOWN, detail=detail)
        for item in items
    )
    return ExecutionOutcome(items=unknown_items)


def _execution_outcome_for_status(
    items: tuple[TransferItem, ...], status: ExecutionStatus, detail: str
) -> ExecutionOutcome:
    item_outcomes = tuple(
        ExecutionItemOutcome(backend_key=item.backend_key, status=status, detail=detail) for item in items
    )
    return ExecutionOutcome(items=item_outcomes)


def _describe_execution_error(stage: str, error: Exception) -> str:
    return f"{stage} raised {type(error).__name__}: {error}"


class OperationResolver:
    """Resolve logical operation requests into worker-local execution tasks."""

    def __init__(self, token_database: ChunkedTokenDatabase) -> None:
        self._token_database = token_database

    def resolve_lookup(self, request: LookupRequest) -> LookupTask:
        content_hashes = self._content_hashes(request.content_ids)
        token_chunks = self._token_database.process_token_key_strings(request.candidate_span.length, content_hashes)

        candidates: list[LookupCandidate] = []
        for start, end, backend_key, _ in token_chunks:
            candidate_span = self._absolute_span(request.candidate_span, start, end)
            candidates.append(LookupCandidate(backend_key=backend_key, token_span=candidate_span))

        return LookupTask(operation_ref=request.operation_ref, candidates=tuple(candidates))

    def resolve_load(self, request: LoadRequest) -> LoadTask:
        destination_items = self._resolve_transfer_items(
            request.load_span, request.content_ids, request.destination_blocks
        )
        return LoadTask(operation_ref=request.operation_ref, destination_items=destination_items)

    def resolve_store(self, request: StoreRequest, source_ready: SourceReady) -> StoreTask:
        source_items = self._resolve_transfer_items(request.store_span, request.content_ids, request.source_blocks)
        return StoreTask(operation_ref=request.operation_ref, source_ready=source_ready, source_items=source_items)

    def _resolve_transfer_items(
        self,
        token_span: TokenSpan,
        content_ids: tuple[ContentId, ...],
        local_blocks: tuple[LocalBlockRef, ...],
    ) -> tuple[TransferItem, ...]:
        content_hashes = self._content_hashes(content_ids)
        block_ids = [block.block_id for block in local_blocks]
        token_chunks = self._token_database.process_token_key_strings_with_block_ids(
            token_span.length, content_hashes, block_ids
        )

        items: list[TransferItem] = []
        for start, end, backend_key, _, block_id in token_chunks:
            addresses, byte_counts, resolved_block_id = self._token_database.prepare_value(
                start, end, block_ids, block_id=block_id
            )
            memory_segments = tuple(
                MemorySegment(address=address, byte_count=byte_count)
                for address, byte_count in zip(addresses, byte_counts, strict=True)
            )
            transfer_item = TransferItem(
                backend_key=backend_key,
                token_span=self._absolute_span(token_span, start, end),
                local_block=LocalBlockRef(resolved_block_id),
                memory_segments=memory_segments,
            )
            items.append(transfer_item)

        return tuple(items)

    @staticmethod
    def _content_hashes(content_ids: tuple[ContentId, ...]) -> list[str]:
        return [content_id.value.hex() for content_id in content_ids]

    @staticmethod
    def _absolute_span(base_span: TokenSpan, start: int, end: int) -> TokenSpan:
        return TokenSpan(base_span.start + start, base_span.start + end)


class LookupExecutor:
    """Worker-local lookup execution boundary."""

    def __init__(self, backend: KVStoreBackend) -> None:
        self._backend = backend

    def execute(self, task: LookupTask) -> LookupExecutionOutcome:
        backend_keys = tuple(candidate.backend_key for candidate in task.candidates)
        try:
            outcome = self._backend.exists(backend_keys)
        except Exception as error:
            return LookupExecutionUnavailable(
                status=ExecutionStatus.UNKNOWN,
                detail=_describe_execution_error("backend lookup", error),
            )

        if isinstance(outcome, LookupExecutionSucceeded) and len(outcome.candidate_hits) != len(backend_keys):
            return LookupExecutionUnavailable(
                status=ExecutionStatus.UNKNOWN,
                detail="backend lookup result count does not match the requested key count",
            )
        return outcome


class LoadExecutor:
    """Worker-local load execution boundary."""

    def __init__(self, backend: KVStoreBackend) -> None:
        self._backend = backend

    def execute(self, task: LoadTask) -> ExecutionOutcome:
        """Return after successful destination items are visible to model execution."""
        try:
            outcome = self._backend.get(task.destination_items)
        except Exception as error:
            detail = _describe_execution_error("backend load", error)
            return _execution_outcome_for_status(task.destination_items, ExecutionStatus.UNKNOWN, detail)

        return _aligned_transfer_outcome_or_unknown(task.destination_items, outcome)


class StoreExecutor:
    """Accept Store tasks and execute them outside the model thread."""

    def __init__(self, backend: KVStoreBackend) -> None:
        self._backend = backend
        self._background_worker = ThreadPoolExecutor(
            max_workers=STORE_BACKGROUND_WORKER_COUNT, thread_name_prefix="ascendstore-store"
        )

    def submit(
        self,
        task: StoreTask,
        notify_source_released: Callable[[OperationRef], None],
        notify_source_release_unknown: Callable[[OperationRef, str], None],
        notify_store_completed: Callable[[OperationRef, ExecutionOutcome], None],
    ) -> None:
        try:
            self._background_worker.submit(
                self._execute,
                task,
                notify_source_released,
                notify_source_release_unknown,
                notify_store_completed,
            )
        except Exception as error:
            detail = _describe_execution_error("store task submission", error)
            outcome = _execution_outcome_for_status(task.source_items, ExecutionStatus.FAILED, detail)
            notify_source_released(task.operation_ref)
            notify_store_completed(task.operation_ref, outcome)

    def close(self) -> None:
        self._background_worker.shutdown()

    def _execute(
        self,
        task: StoreTask,
        notify_source_released: Callable[[OperationRef], None],
        notify_source_release_unknown: Callable[[OperationRef, str], None],
        notify_store_completed: Callable[[OperationRef, ExecutionOutcome], None],
    ) -> None:
        try:
            task.source_ready.wait()
        except Exception as error:
            detail = _describe_execution_error("source readiness wait", error)
            outcome = _execution_outcome_for_status(task.source_items, ExecutionStatus.FAILED, detail)
            notify_source_released(task.operation_ref)
            notify_store_completed(task.operation_ref, outcome)
            return

        try:
            backend_outcome = self._backend.put(task.source_items)
        except Exception as error:
            detail = _describe_execution_error("backend store", error)
            outcome = _execution_outcome_for_status(task.source_items, ExecutionStatus.UNKNOWN, detail)
            notify_source_release_unknown(task.operation_ref, detail)
            notify_store_completed(task.operation_ref, outcome)
            return

        outcome = _aligned_transfer_outcome_or_unknown(task.source_items, backend_outcome)

        notify_source_released(task.operation_ref)
        notify_store_completed(task.operation_ref, outcome)
