"""Observed facts produced after selected KV Pool work begins execution."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash

from ...coordinates import TokenRange
from ...protocol.lookup import TailKeyBoundary
from .representation import RemoteKVObject
from .transfer import TransferSource


@dataclass(frozen=True, slots=True)
class ChunkAvailability:
    """Backend availability observed for one semantic KV chunk."""

    token_range: TokenRange
    content_hash: BlockHash | str
    available: bool


@dataclass(frozen=True, slots=True)
class GroupAvailability:
    """Semantic chunk observations for one original vLLM cache group."""

    group_id: int
    chunks: tuple[ChunkAvailability, ...]


@dataclass(frozen=True, slots=True)
class ReachablePrefix:
    """Common reachable frontier and the remote identities needed to load its tail."""

    end_token: int
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


@dataclass(frozen=True, slots=True)
class RemoteObjectObservation:
    """Readability observed for one exact remote representation."""

    remote_object: RemoteKVObject
    readable: bool


@dataclass(frozen=True, slots=True)
class TransferEvidence:
    """Backend-normalized result code attached to one submitted request row."""

    source: TransferSource
    result_code: int | None
    source_release_confirmed: bool | None = None


@dataclass(frozen=True, slots=True)
class StoreEvidence:
    """Authoritative outcome at the producer's current Store execution granularity.

    Per-row codes retain diagnostics; consumers use the success summary rather than
    deriving it again. A layer range can succeed before its remote object is committed;
    session finalization supplies the whole-transfer outcome consumed by Runtime.
    Source release independently confirms this operation no longer reads its source rows.
    """

    transfer_evidence: tuple[TransferEvidence, ...]
    succeeded: bool
    source_release_confirmed: bool
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class LoadCompletion:
    """Load evidence published when one selected transfer completes."""

    request_id: str
    transfer_evidence: tuple[TransferEvidence, ...]


@dataclass(frozen=True, slots=True)
class LoadFailure:
    """Load failure facts reduced from one completed transfer."""

    failed_request_ids: frozenset[str] = frozenset()
    failed_block_ids: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class StoreCompletion:
    """Evidence for one executed transfer, which may be only one layer range."""

    request_id: str
    evidence: StoreEvidence
    store_job_id: int | None = None
