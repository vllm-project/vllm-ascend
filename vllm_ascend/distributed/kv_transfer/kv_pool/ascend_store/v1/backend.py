"""Backend boundaries for the AscendStore v1 classic path."""

from typing import Protocol

from .protocol import ExecutionOutcome, LookupExecutionOutcome, TransferItem


class KVStoreBackend(Protocol):
    """Backend-neutral boundary that translates SDK return codes and exceptions into explicit outcomes."""

    def exists(self, backend_keys: tuple[str, ...]) -> LookupExecutionOutcome: ...

    def get(self, items: tuple[TransferItem, ...]) -> ExecutionOutcome:
        """Return ordered per-item outcomes after successful destinations are ready for model execution."""
        ...

    def put(self, items: tuple[TransferItem, ...]) -> ExecutionOutcome:
        """Return ordered per-item outcomes after the backend no longer accesses source memory."""
        ...
