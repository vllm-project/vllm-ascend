"""External-storage contracts, independent of requests, KV layouts, and layer scheduling.

Callers provide ready, registered local memory and keep it alive through copying.
They serialize operations on each object's session; these protocols do not add
thread safety to a native client. Read and write sessions have separate lifetimes.
Per-key results preserve input order: zero means success, nonzero means failure
or unconfirmed cleanup. Invalid native results raise rather than imply success.

Backend session owners retain bookkeeping across native calls. A failed cleanup
must not discard that ownership or authorize source-memory reuse. SDK adapters
own clients and registration mechanics, not request completion decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

from .registration import Registration

AddressRows: TypeAlias = list[list[int]]
SizeRows: TypeAlias = list[list[int]]
OffsetRows: TypeAlias = list[list[int]]
ResultCodes: TypeAlias = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GVARegion:
    """One contiguous remote object; an address snapshot does not itself pin it."""

    address: int
    size: int


class ObjectSessions(Protocol):
    """Backend-owned remote resources; callers schedule operations after copy fences.

    Keys are opaque object identities, not requests. Each direction is serialized
    by its caller. Failed release retains ownership, including uncertain opens.
    close runs after both directions drain and raises while cleanup is unresolved.
    """

    @property
    def has_load_sessions(self) -> bool: ...

    @property
    def pending_store_keys(self) -> tuple[str, ...]: ...

    def start_load(self, keys: list[str], object_sizes: list[int]) -> ResultCodes: ...

    def finish_load(self, keys: list[str] | None = None) -> None: ...

    def start_store(self, keys: list[str], object_sizes: list[int]) -> ResultCodes: ...

    def commit_store(self, keys: list[str]) -> ResultCodes:
        """Publish caller-selected complete objects; retain unsuccessful sessions."""
        ...

    def revoke_store(self, keys: list[str]) -> ResultCodes:
        """Abandon owned writes after copies stop; only zero confirms remote cleanup.

        Unsupported safe revocation returns nonzero and retains the ownership
        record. Neither revocation nor publication is a local-memory copy fence.
        """
        ...

    def close(self) -> None: ...


class GVAObjectSessions(ObjectSessions, Protocol):
    """Expose addresses only while the corresponding remote sessions remain usable.

    Region snapshots do not pin objects; keep the owner alive and finish copying
    before releasing or committing its sessions.
    """

    def load_regions(self, keys: tuple[str, ...], object_size: int) -> tuple[GVARegion, ...]: ...

    def store_regions(self, keys: tuple[str, ...], object_size: int) -> tuple[GVARegion, ...]: ...


class KVStoreBackend(Protocol):
    """Lifecycle and whole-object operations required by every v1 Backend."""

    requires_exists_before_put: bool

    def set_device(self) -> None:
        """Initialize the calling execution thread's device context on an open client."""
        ...

    def register_buffer(self, addresses: list[int], sizes: list[int]) -> Registration:
        """Return a registration owner; incomplete rollback exposes a retryable owner.

        A lazy adapter may defer native registration until initialization, but
        must complete it before copying. Keep this owner until all copies stop.
        """
        ...

    def close(self) -> None:
        """Close the local client after callers drain copies and release registrations.

        Successful close is idempotent; failed close retains state for retry.
        This is not a copy fence or proof that abandoned remote writes expired.
        Other operations must reject a successfully closed client.
        """
        ...

    def exists(self, keys: list[str]) -> tuple[bool, ...]:
        """Return readable-object presence, not a lease or reservation against eviction."""
        ...

    def load(self, keys: list[str], addresses: AddressRows, sizes: SizeRows) -> ResultCodes:
        """Finish destination writes before normal return; an exception is not a fence."""
        ...

    def store(self, keys: list[str], addresses: AddressRows, sizes: SizeRows) -> ResultCodes:
        """Finish all source reads before returning per-key success or failure codes.

        An exception does not confirm source release. A failed cache write must
        not make an incomplete object readable through ``exists`` or ``load``.
        """
        ...


class KeyRangeBackend(KVStoreBackend, Protocol):
    """Copy object-key byte ranges under Backend-owned sessions.

    The Worker chooses keys and offsets and decides which complete objects to
    commit. Normal copy return ends local-memory access, including failed copies;
    zero per key means every requested byte was copied. Native session mechanics
    are internal to the selected Backend.
    """

    def create_sessions(self) -> ObjectSessions:
        """Validate support and return the owner retained until copies and cleanup finish."""
        ...

    def copy_key_range_load(
        self,
        keys: list[str],
        addresses: AddressRows,
        sizes: SizeRows,
        source_offsets: OffsetRows,
    ) -> ResultCodes: ...

    def copy_key_range_store(
        self,
        keys: list[str],
        addresses: AddressRows,
        sizes: SizeRows,
        destination_offsets: OffsetRows,
    ) -> ResultCodes: ...


class GVABackend(KVStoreBackend, Protocol):
    """Copy global-address ranges while a Backend-owned session pins their regions.

    The Worker prepares addresses from the session's regions. Lease acquisition,
    allocation, publication, and remote cleanup belong to the implementation.
    Publication and completion of local source reads are separate guarantees.
    """

    def create_sessions(self) -> GVAObjectSessions:
        """Validate support and return the owner retained until copies and cleanup finish."""
        ...

    def query_gva_regions(self, keys: list[str]) -> tuple[GVARegion | None, ...]:
        """Return readable snapshots for probing; use session regions for copying."""
        ...

    def copy_from_gva(
        self,
        remote_addresses: list[int],
        local_addresses: list[int],
        sizes: list[int],
    ) -> int:
        """Finish destination writes before normal return; the status covers the whole batch."""
        ...

    def copy_to_gva(
        self,
        remote_addresses: list[int],
        local_addresses: list[int],
        sizes: list[int],
    ) -> int:
        """Finish all source reads before returning a success or failure code."""
        ...
