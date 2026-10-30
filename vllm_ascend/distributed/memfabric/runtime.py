# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local claims for MemFabric API families with native singletons."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from threading import Lock, RLock

MEMFABRIC_API_OFFLOAD = "offload"
MEMFABRIC_API_BM = "bm"
MEMFABRIC_API_MEMCACHE = "memcache"
MEMFABRIC_API_TRANSFER_ENGINE = "transfer_engine"


@dataclass
class _ActiveClaim:
    owner: str
    signature: Hashable
    references: int


class MemFabricRuntimeRegistry:
    """Reject incompatible clients before they initialize a native singleton.

    One registry instance must be shared by clients in a process. API-family
    keys distinguish separate MemFabric interfaces such as ``offload``,
    ``bm``, and ``transfer_engine``. A family may have only one logical owner
    and configuration signature at a time; repeated acquisitions by that same
    owner/configuration are reference-counted.

    The registry does not initialize, uninitialize, or otherwise call native
    MemFabric APIs. A client must release its lease only after its backend has
    synchronized and closed its resources.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._claims: dict[str, _ActiveClaim] = {}

    def acquire(self, api_family: str, owner: str, signature: Hashable) -> MemFabricRuntimeLease:
        """Claim an API family or fail before native initialization."""
        if not api_family:
            raise ValueError("api_family must not be empty")
        if not owner:
            raise ValueError("owner must not be empty")
        try:
            hash(signature)
        except TypeError as exc:
            raise TypeError("signature must be hashable and stable for the resource lifetime") from exc

        with self._lock:
            active = self._claims.get(api_family)
            if active is None:
                self._claims[api_family] = _ActiveClaim(owner, signature, 1)
            elif (active.owner, active.signature) == (owner, signature):
                active.references += 1
            else:
                raise RuntimeError(
                    f"MemFabric API family {api_family!r} is already claimed by "
                    f"owner={active.owner!r}; cannot acquire for owner={owner!r}"
                )
        return MemFabricRuntimeLease(self, api_family, owner, signature)

    def _release(self, api_family: str, owner: str, signature: Hashable) -> None:
        with self._lock:
            active = self._claims.get(api_family)
            if active is None or (active.owner, active.signature) != (owner, signature):
                raise RuntimeError(f"No matching MemFabric runtime claim exists for API family {api_family!r}")
            active.references -= 1
            if active.references == 0:
                del self._claims[api_family]

    def active_claims(self) -> tuple[tuple[str, str, Hashable, int], ...]:
        """Return a stable snapshot for diagnostics and startup error reports."""
        with self._lock:
            return tuple(
                (api_family, claim.owner, claim.signature, claim.references)
                for api_family, claim in sorted(self._claims.items())
            )


class MemFabricRuntimeLease:
    """Idempotently release one acquisition from a runtime claim registry."""

    def __init__(
        self,
        registry: MemFabricRuntimeRegistry,
        api_family: str,
        owner: str,
        signature: Hashable,
    ) -> None:
        self._registry = registry
        self._api_family = api_family
        self._owner = owner
        self._signature = signature
        self._released = False
        self._lock = Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._registry._release(self._api_family, self._owner, self._signature)
            self._released = True

    def __enter__(self) -> MemFabricRuntimeLease:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


# The registry exists for the process lifetime. Clients retain their leases
# alongside the native runtime and release only after synchronizing and closing it.
global_memfabric_runtime_registry = MemFabricRuntimeRegistry()
