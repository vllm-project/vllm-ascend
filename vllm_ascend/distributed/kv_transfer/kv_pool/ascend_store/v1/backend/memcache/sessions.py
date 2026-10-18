"""Own Memcache object sessions without Worker or KV-layout dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..interfaces import GVARegion
from ..results import integer_results
from ..validation import validate_session_open

if TYPE_CHECKING:
    from .backend import MemcacheBackend

GVA_SESSION_FAILURE = -1


class MemcacheSessions:
    """Retain remote ownership until explicit release, publication, or revocation.

    Read and write dictionaries belong to separate serial executors; close runs
    only after both drain. None denotes an unresolved acquisition or abandoned
    write, never an address that may be copied. Cleanup failure retains the key
    and prevents reopening it; server expiry is not a local release confirmation.
    """

    def __init__(self, backend: MemcacheBackend) -> None:
        backend.validate_gva_support()
        self._backend = backend
        self._load_regions: dict[str, GVARegion | None] = {}
        self._store_regions: dict[str, GVARegion | None] = {}

    @property
    def has_load_sessions(self) -> bool:
        return bool(self._load_regions)

    @property
    def pending_store_keys(self) -> tuple[str, ...]:
        return tuple(self._store_regions)

    # ====================
    # Read leases
    # ====================

    def start_load(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        validate_session_open(keys, object_sizes, self._load_regions)
        if not keys:
            return ()
        self._load_regions.update((key, None) for key in keys)
        codes = list(integer_results("GVA acquire leases", keys, self._backend.add_gva_leases(keys)))
        for key, code in zip(keys, codes, strict=True):
            if code != 0:
                del self._load_regions[key]
        leased = [key for key, code in zip(keys, codes, strict=True) if code == 0]
        # Lookup addresses can have expired. Resolve only after the new lease is held.
        regions = dict(zip(leased, self._backend.query_gva_regions(leased), strict=True)) if leased else {}
        rejected = []
        for index, (key, size) in enumerate(zip(keys, object_sizes, strict=True)):
            if codes[index] != 0:
                continue
            region = regions[key]
            if region is None or region.size != size:
                codes[index] = GVA_SESSION_FAILURE
                rejected.append(key)
            else:
                self._load_regions[key] = region
        self.finish_load(rejected)
        return tuple(codes)

    def load_regions(self, keys: tuple[str, ...], object_size: int) -> tuple[GVARegion, ...]:
        return self._resolve_regions(keys, object_size, self._load_regions)

    def finish_load(self, keys: list[str] | None = None) -> None:
        owned = (
            list(self._load_regions)
            if keys is None
            else list(dict.fromkeys(key for key in keys if key in self._load_regions))
        )
        if owned:
            # A failed batch release may already have dropped a subset of leases.
            self._load_regions.update((key, None) for key in owned)
            self._backend.remove_gva_leases(owned)
            for key in owned:
                del self._load_regions[key]

    # ====================
    # Write allocations
    # ====================

    def start_store(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        validate_session_open(keys, object_sizes, self._store_regions)
        if not keys:
            return ()
        self._store_regions.update((key, None) for key in keys)
        addresses = integer_results("GVA allocate", keys, self._backend.allocate_gva(keys, object_sizes))
        codes = []
        for key, address, size in zip(keys, addresses, object_sizes, strict=True):
            codes.append(0 if address > 0 else GVA_SESSION_FAILURE)
            if address > 0:
                self._store_regions[key] = GVARegion(address, size)
            else:
                del self._store_regions[key]
        return tuple(codes)

    def store_regions(self, keys: tuple[str, ...], object_size: int) -> tuple[GVARegion, ...]:
        return self._resolve_regions(keys, object_size, self._store_regions)

    def commit_store(self, keys: list[str]) -> tuple[int, ...]:
        if not keys:
            return ()
        if len(set(keys)) != len(keys):
            raise ValueError("Commit keys must be unique")
        if any(self._store_regions.get(key) is None for key in keys):
            raise RuntimeError("GVA commit has no confirmed owning Store session")
        codes = integer_results("GVA publish", keys, self._backend.publish_gva(keys))
        for key, code in zip(keys, codes, strict=True):
            if code == 0:
                del self._store_regions[key]
        return codes

    def revoke_store(self, keys: list[str]) -> tuple[int, ...]:
        owned = list(dict.fromkeys(key for key in keys if key in self._store_regions))
        if not owned:
            return (0,) * len(keys)
        # Stop exposing addresses even if the adapter cannot safely release them yet.
        self._store_regions.update((key, None) for key in owned)
        codes = integer_results("GVA revoke", owned, self._backend.revoke_gva(owned))
        by_key = dict(zip(owned, codes, strict=True))
        for key, code in by_key.items():
            if code == 0:
                del self._store_regions[key]
        return tuple(by_key.get(key, 0) for key in keys)

    def close(self) -> None:
        """Retry cleanup without dropping records or claiming that server expiry occurred."""
        self.finish_load()
        self.revoke_store(list(self._store_regions))
        if self._store_regions:
            raise RuntimeError(f"GVA Store cleanup is unconfirmed for {tuple(self._store_regions)}")

    @staticmethod
    def _resolve_regions(
        keys: tuple[str, ...], object_size: int, regions: dict[str, GVARegion | None]
    ) -> tuple[GVARegion, ...]:
        resolved = []
        for key in keys:
            region = regions.get(key)
            if region is None:
                raise RuntimeError(f"GVA session for {key!r} is unavailable")
            if region.size != object_size:
                raise RuntimeError(f"GVA session for {key!r} has an unexpected object size")
            resolved.append(region)
        return tuple(resolved)
