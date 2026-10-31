# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Caller-owned ID planning for MemFabric Offload's native pools."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

# MemFabric Offload supports IDs 0..63. Keep 0 for the default-pool client.
_OFFLOAD_POOL_COUNT = 64


@dataclass(frozen=True, slots=True)
class OffloadPoolBinding:
    """One owner's native pool selector and dedicated SHARED rendezvous URL.

    This is startup metadata, not a native handle or a reference-counted lease.
    The owner passes ``pool_id`` to every Offload call and ``store_url`` to its
    OffloadConfig. Only that owner closes the pool, after stopping graph replay,
    synchronizing its streams and releasing all views. Keep the binding fixed
    until teardown; a reused ID does not make old addresses or graphs valid.
    """

    owner: str
    pool_id: int
    store_url: str

    def __post_init__(self) -> None:
        if not isinstance(self.owner, str) or not self.owner:
            raise ValueError("owner must be a non-empty string")
        if isinstance(self.pool_id, bool) or not isinstance(self.pool_id, Integral):
            raise ValueError("pool_id must be an integer in [0, 64)")
        if not 0 <= self.pool_id < _OFFLOAD_POOL_COUNT:
            raise ValueError("pool_id must be an integer in [0, 64)")
        if not isinstance(self.store_url, str) or not self.store_url.startswith("tcp://"):
            raise ValueError("store_url must be a dedicated tcp:// endpoint")
        object.__setattr__(self, "pool_id", int(self.pool_id))


def plan_offload_pools(
    store_urls: Mapping[str, str], *, default_pool_owner: str | None = None
) -> dict[str, OffloadPoolBinding]:
    """Assign IDs from a complete job-wide owner-to-endpoint configuration.

    Build once before worker initialization and distribute the bindings to all
    participating ranks, including ranks that use only a subset of the pools.
    Sorting owner names makes IDs independent of initialization order. Never
    recompute a plan from rank-local subsets or while any pool is still active.

    An existing default-pool client may be named by ``default_pool_owner``; it
    keeps ID 0. Otherwise 0 stays reserved. Other owners receive IDs from 1.
    Each SHARED pool needs its own endpoint, even if its membership is identical.
    Supply canonical URLs; string comparison cannot detect host/IP aliases.
    Membership, capacity, device and rank remain native OffloadConfig fields.
    MemCache and TransferEngine use separate APIs and do not consume this plan.
    """
    if default_pool_owner is not None and default_pool_owner not in store_urls:
        raise ValueError("default_pool_owner must be present in store_urls")
    owners = [owner for owner in store_urls if owner != default_pool_owner]
    if len(owners) >= _OFFLOAD_POOL_COUNT:
        raise ValueError("at most 63 non-default Offload pools are supported")
    if any(not isinstance(owner, str) or not owner for owner in store_urls):
        raise ValueError("owner must be a non-empty string")

    bindings = {
        owner: OffloadPoolBinding(owner, pool_id, store_urls[owner])
        for pool_id, owner in enumerate(sorted(owners), start=1)
    }
    if default_pool_owner is not None:
        bindings[default_pool_owner] = OffloadPoolBinding(default_pool_owner, 0, store_urls[default_pool_owner])
    if len({binding.store_url for binding in bindings.values()}) != len(bindings):
        raise ValueError("each Offload pool must have a distinct store_url")
    return bindings
