"""Backend evidence consumed by Runtime and Timeline state."""

from __future__ import annotations

from dataclasses import dataclass

from .batch import TransferSource


@dataclass(frozen=True, slots=True)
class TransferEvidence:
    source: TransferSource
    result_code: int | None
    source_release_confirmed: bool | None = None


@dataclass(frozen=True, slots=True)
class StoreEvidence:
    transfer_evidence: tuple[TransferEvidence, ...]
    succeeded: bool
    source_release_confirmed: bool
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class LoadCompletion:
    request_id: str
    transfer_evidence: tuple[TransferEvidence, ...]


@dataclass(frozen=True, slots=True)
class StoreCompletion:
    request_id: str
    evidence: StoreEvidence
    store_job_id: int | None = None
