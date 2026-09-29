"""Request-local values and the resumable frame of one KV Pool program invocation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..protocol.transfer import KVTransferStep
from .representation import BindingBatch, KVBinding, RemoteKVObject

# ===============================
# Invocation Values
# ===============================


@dataclass(frozen=True, slots=True)
class RemoteObjectObservation:
    """Readability reported for one exact remote representation."""

    remote_object: RemoteKVObject
    readable: bool


@dataclass(frozen=True, slots=True)
class BindingEvidence:
    """Native result code aligned with one exact transfer edge."""

    binding: KVBinding
    result_code: int | None


@dataclass(frozen=True, slots=True)
class StoreEvidence:
    """Store execution evidence with source-release kept independent."""

    binding_evidence: tuple[BindingEvidence, ...]
    succeeded: bool
    source_release_confirmed: bool
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class LoadTransfer:
    """One request's immutable Load work after spatial projection."""

    request_id: str
    traversal: tuple[KVBinding, ...]


@dataclass(frozen=True, slots=True)
class LoadCompletion:
    """Load evidence produced when one transfer reaches its completion point."""

    request_id: str
    binding_evidence: tuple[BindingEvidence, ...]


@dataclass(frozen=True, slots=True)
class StoreTransfer:
    """One request's immutable Store work after spatial projection."""

    request_id: str
    batches: tuple[BindingBatch, ...]


@dataclass(frozen=True, slots=True)
class StoreCompletion:
    """Store completion with transfer and source-release facts kept separate."""

    request_id: str
    evidence: StoreEvidence


# ===============================
# Invocation Frame
# ===============================


@dataclass(slots=True)
class KVPoolStepFrame:
    """Hold request-local state across one invocation of ``KVPoolProgram``."""

    step: KVTransferStep
    failed_request_ids: set[str] = field(default_factory=set)
    failed_block_ids: set[int] = field(default_factory=set)
    store_transfers: list[StoreTransfer] = field(default_factory=list)
