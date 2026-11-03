# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Caller-owned instance bindings for MemFabric Offload."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral


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
            raise ValueError("initialization_order must be a non-negative integer")
        if self.initialization_order < 0:
            raise ValueError("initialization_order must be a non-negative integer")
        object.__setattr__(self, "initialization_order", int(self.initialization_order))


def plan_offload_instances(
    store_urls: Mapping[str, str], *, reserved_instances: int = 0
) -> dict[str, OffloadInstanceBinding]:
    """Build stable logical-owner bindings for independent Offload instances.

    Build once from the complete job-wide configuration and distribute the
    owner-to-endpoint metadata to participating ranks. Each rank creates its
    own ``offload.Offload()`` object for each planned owner and initializes
    those objects in ``initialization_order`` on every participating rank; the
    native library assigns the per-process entity ID. Sorting owners makes the
    plan independent of input mapping order. ``reserved_instances`` counts
    Offload instances initialized outside this plan, such as the module-level
    default instance; initialize them first and in the same order on relevant
    ranks. A rank must not skip an earlier
    instance and then join a later SHARED instance, because the resulting
    native entity IDs would differ across ranks.  The order also includes any
    instance count reserved before these bindings.
    The default module-level Offload client is outside this plan and remains
    available for legacy callers.

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
    if len(store_urls) + reserved_instances > 63:
        raise ValueError("at most 63 Offload instances may be initialized per process")
    if len(set(store_urls.values())) != len(store_urls):
        raise ValueError("each SHARED Offload instance must have a distinct store_url")

    return {
        owner: OffloadInstanceBinding(owner, store_urls[owner], reserved_instances + initialization_order)
        for initialization_order, owner in enumerate(sorted(store_urls))
    }
