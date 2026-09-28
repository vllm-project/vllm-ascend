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
