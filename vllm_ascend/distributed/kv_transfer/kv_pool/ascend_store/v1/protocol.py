"""Operation data contracts for the AscendStore v1 classic path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

# ====================
# Shared Values
# ====================


@dataclass(frozen=True, slots=True)
class OperationRef:
    """Identify one KV Pool operation and its owning vLLM request."""

    operation_id: str
    request_id: str

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id must not be empty")
        if not self.request_id:
            raise ValueError("request_id must not be empty")


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """Represent a half-open token range."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("token span start must be non-negative")
        if self.end < self.start:
            raise ValueError("token span end must not precede start")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class ContentId:
    """Identify content without encoding a backend storage key."""

    value: bytes

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("content identity must not be empty")


@dataclass(frozen=True, slots=True)
class LocalBlockRef:
    """Refer to a worker-local logical block without exposing its address."""

    block_id: int

    def __post_init__(self) -> None:
        if self.block_id < 0:
            raise ValueError("local block id must be non-negative")


@dataclass(frozen=True, slots=True)
class MemorySegment:
    """Describe one contiguous worker-local memory range."""

    address: int
    byte_count: int

    def __post_init__(self) -> None:
        if self.address < 0:
            raise ValueError("segment address must be non-negative")
        if self.byte_count <= 0:
            raise ValueError("segment byte count must be positive")


@dataclass(frozen=True, slots=True)
class TransferItem:
    """Bind one token range and backend object to worker-local memory."""

    backend_key: str
    token_span: TokenSpan
    local_block: LocalBlockRef
    memory_segments: tuple[MemorySegment, ...]

    def __post_init__(self) -> None:
        if not self.backend_key:
            raise ValueError("backend key must not be empty")
        if not self.memory_segments:
            raise ValueError("transfer item must contain at least one memory segment")


class ExecutionStatus(str, Enum):
    """Describe whether an attempted backend item has a known outcome."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ExecutionItemOutcome:
    """Record the outcome reported for one backend key."""

    backend_key: str
    status: ExecutionStatus
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.backend_key:
            raise ValueError("backend key must not be empty")


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """Preserve ordered per-key execution outcomes without hiding partial success."""

    items: tuple[ExecutionItemOutcome, ...]

    @property
    def status(self) -> ExecutionStatus:
        if any(item.status is ExecutionStatus.UNKNOWN for item in self.items):
            return ExecutionStatus.UNKNOWN
        if any(item.status is ExecutionStatus.FAILED for item in self.items):
            return ExecutionStatus.FAILED
        return ExecutionStatus.SUCCEEDED


# ====================
# Lookup Operation
# ====================


@dataclass(frozen=True, slots=True)
class LookupRequest:
    """Describe ordered content candidates covering candidate_span."""

    operation_ref: OperationRef
    candidate_span: TokenSpan
    content_ids: tuple[ContentId, ...]


@dataclass(frozen=True, slots=True)
class LookupCandidate:
    backend_key: str
    token_span: TokenSpan

    def __post_init__(self) -> None:
        if not self.backend_key:
            raise ValueError("backend key must not be empty")


@dataclass(frozen=True, slots=True)
class LookupTask:
    operation_ref: OperationRef
    candidates: tuple[LookupCandidate, ...]


@dataclass(frozen=True, slots=True)
class LookupExecutionSucceeded:
    candidate_hits: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class LookupExecutionUnavailable:
    status: ExecutionStatus
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status is ExecutionStatus.SUCCEEDED:
            raise ValueError("an unavailable lookup execution cannot be successful")


LookupExecutionOutcome: TypeAlias = LookupExecutionSucceeded | LookupExecutionUnavailable


@dataclass(frozen=True, slots=True)
class LookupResult:
    operation_ref: OperationRef
    observed_prefix_end: int

    def __post_init__(self) -> None:
        if self.observed_prefix_end < 0:
            raise ValueError("observed prefix end must be non-negative")


@dataclass(frozen=True, slots=True)
class LookupUnavailable:
    operation_ref: OperationRef
    status: ExecutionStatus
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status is ExecutionStatus.SUCCEEDED:
            raise ValueError("an unavailable lookup cannot be successful")


LookupResponse: TypeAlias = LookupResult | LookupUnavailable


# ====================
# Load Operation
# ====================


@dataclass(frozen=True, slots=True)
class LoadRequest:
    """Bind ordered content identities to destination blocks over load_span."""

    operation_ref: OperationRef
    load_span: TokenSpan
    content_ids: tuple[ContentId, ...]
    destination_blocks: tuple[LocalBlockRef, ...]


@dataclass(frozen=True, slots=True)
class LoadTask:
    operation_ref: OperationRef
    destination_items: tuple[TransferItem, ...]


@dataclass(frozen=True, slots=True)
class LoadResult:
    operation_ref: OperationRef
    ready_prefix_end: int
    invalid_destination_blocks: tuple[LocalBlockRef, ...]
    outcome: ExecutionOutcome

    def __post_init__(self) -> None:
        if self.ready_prefix_end < 0:
            raise ValueError("ready prefix end must be non-negative")


# ====================
# Store Operation
# ====================


class SourceReady(Protocol):
    """Wait until an asynchronous Store may safely read its source blocks."""

    def wait(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StoreRequest:
    """Bind ordered content identities to source blocks over store_span."""

    operation_ref: OperationRef
    store_span: TokenSpan
    content_ids: tuple[ContentId, ...]
    source_blocks: tuple[LocalBlockRef, ...]


@dataclass(frozen=True, slots=True)
class StoreTask:
    operation_ref: OperationRef
    source_ready: SourceReady
    source_items: tuple[TransferItem, ...]


@dataclass(frozen=True, slots=True)
class StoreAccepted:
    operation_ref: OperationRef


@dataclass(frozen=True, slots=True)
class SourceReleased:
    operation_ref: OperationRef


@dataclass(frozen=True, slots=True)
class SourceReleaseUnknown:
    operation_ref: OperationRef
    detail: str

    def __post_init__(self) -> None:
        if not self.detail:
            raise ValueError("unknown source release must include detail")


@dataclass(frozen=True, slots=True)
class StoreResult:
    operation_ref: OperationRef
    outcome: ExecutionOutcome
