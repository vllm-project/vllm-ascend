# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Caller-owned instance bindings for MemFabric Offload."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

# MemFabric reserves a 64-entry native entity-ID segment for Offload.
MAX_OFFLOAD_ENTITY_IDS = 64


@dataclass(frozen=True, slots=True)
class OffloadInstanceBinding:
    """One logical owner's Offload instance configuration.

    The binding is startup metadata, not a native handle. The adapter creates
    one ``offload.Offload()`` object for the owner and initializes it with this
    binding's ``store_url``. The native instance owns its entity ID and
    lifecycle; callers must not allocate or pass an ID through the data path.
    ``initialization_order`` keeps entity-ID assignment consistent across
    ranks that participate in the same SHARED instances.
    """

    owner: str
    store_url: str
    initialization_order: int

    def __post_init__(self) -> None:
        if not isinstance(self.owner, str) or not self.owner:
            raise ValueError("owner must be a non-empty string")
        if not isinstance(self.store_url, str) or not self.store_url.startswith("tcp://"):
            raise ValueError("store_url must be a dedicated tcp:// endpoint")
        if isinstance(self.initialization_order, bool) or not isinstance(self.initialization_order, Integral):
            raise ValueError(f"initialization_order must be an integer in [0, {MAX_OFFLOAD_ENTITY_IDS})")
        if not 0 <= self.initialization_order < MAX_OFFLOAD_ENTITY_IDS:
            raise ValueError(f"initialization_order must be an integer in [0, {MAX_OFFLOAD_ENTITY_IDS})")
        object.__setattr__(self, "initialization_order", int(self.initialization_order))


def plan_offload_instances(
    store_urls: Mapping[str, str], *, reserved_instances: int = 0
) -> dict[str, OffloadInstanceBinding]:
    """Build stable logical-owner bindings for independent Offload instances.

    Build once from the complete job-wide configuration and distribute the
    owner-to-endpoint metadata to participating ranks. Each rank creates its
    own ``offload.Offload()`` object for each planned owner and initializes
    those objects in ``initialization_order`` on every participating rank; the
    native library assigns the per-process entity ID. This helper produces
    metadata only: it does not initialize instances, discover rank membership,
    or coordinate lifecycle. Sorting owners makes the plan independent of
    input mapping order. ``reserved_instances`` counts entity IDs consumed
    outside this plan, including the module-level default instance and failed
    initialization attempts; initialize them first and in the same order on
    relevant ranks. A rank must not skip an earlier instance and then join a
    later SHARED instance, because native entity IDs would differ across
    ranks. Planned ``initialization_order`` values start after
    ``reserved_instances``. The default module-level client remains available
    for legacy callers.

    SHARED instances require distinct, deployment-canonical store URLs. The
    planner cannot detect hostname/IP aliases that compare differently as
    strings. MemCache and TransferEngine use separate APIs and do not consume
    this plan.
    """
    if any(not isinstance(owner, str) or not owner for owner in store_urls):
        raise ValueError("owner must be a non-empty string")
    if any(not isinstance(store_url, str) or not store_url.startswith("tcp://") for store_url in store_urls.values()):
        raise ValueError("store_url must be a dedicated tcp:// endpoint")
    if isinstance(reserved_instances, bool) or not isinstance(reserved_instances, Integral) or reserved_instances < 0:
        raise ValueError("reserved_instances must be a non-negative integer")
    reserved_instances = int(reserved_instances)
    if len(store_urls) + reserved_instances > MAX_OFFLOAD_ENTITY_IDS:
        raise ValueError(f"at most {MAX_OFFLOAD_ENTITY_IDS} Offload entity IDs may be consumed per process")
    if len(set(store_urls.values())) != len(store_urls):
        raise ValueError("each SHARED Offload instance must have a distinct store_url")

    return {
        owner: OffloadInstanceBinding(owner, store_urls[owner], reserved_instances + initialization_order)
        for initialization_order, owner in enumerate(sorted(store_urls))
    }
