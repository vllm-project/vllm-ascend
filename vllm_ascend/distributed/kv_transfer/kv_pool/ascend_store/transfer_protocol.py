from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TransferOperation(str, Enum):
    LOAD = "load"
    SAVE = "save"
    LOOKUP = "lookup"
    CANCEL = "cancel"
    SHUTDOWN = "shutdown"


class TransferStatus(str, Enum):
    NEW = "new"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    RETRYABLE_ERROR = "retryable_error"
    FATAL_ERROR = "fatal_error"
    CANCELLED = "cancelled"
    LOST = "lost"


@dataclass(frozen=True)
class TransferCommand:
    """Versioned control-plane command shared by thread and process backends."""

    command_id: int
    request_id: str
    operation: TransferOperation
    deadline_ns: int
    attempt: int = 0
    idempotency_key: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    protocol_version: int = 1


@dataclass(frozen=True)
class TransferResult:
    command_id: int
    request_id: str
    status: TransferStatus
    error_message: str | None = None
    backend_error_code: int | None = None
    retryable: bool = False
    completed_blocks: tuple[int, ...] = ()
    failed_blocks: tuple[int, ...] = ()
    worker_generation: int = 0
    protocol_version: int = 1
