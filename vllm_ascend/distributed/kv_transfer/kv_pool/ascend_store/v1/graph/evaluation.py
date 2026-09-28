"""One resumable evaluation of the fixed KV Pool graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..protocol.transfer import KVTransferStep

if TYPE_CHECKING:
    from ..execution.timeline import StoreBatch


@dataclass(slots=True)
class KVPoolStepEvaluation:
    """Hold the request-local frame for one execution of ``KVPoolGraph``."""

    step: KVTransferStep
    failed_request_ids: set[str] = field(default_factory=set)
    failed_block_ids: set[int] = field(default_factory=set)
    pending_store: StoreBatch | None = None


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Terminal Load facts consumed through vLLM's split completion hooks."""

    completed_request_ids: frozenset[str]
    failed_request_ids: frozenset[str]
    failed_block_ids: frozenset[int]
