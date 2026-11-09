# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Data-plane strategies for layerwise KV pooling.

The layerwise worker drives every backend plane (block-key, GVA) through one
interface: pool-session preparation before the per-layer transfer loop, GVA
target preparation after it, and load-session cleanup at the step boundary.
The plane implementations own the plane-specific protocol; the worker keeps
the shared machinery (key construction, store I/O helpers, session tracking)
both planes call back into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes.base import (
    LayerwiseDataPlane,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes.block_key import (
    BlockKeyDataPlane,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes.gva import (
    GvaDataPlane,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import (
        LayerwiseKVPoolWorker,
    )

__all__ = [
    "BlockKeyDataPlane",
    "GvaDataPlane",
    "LayerwiseDataPlane",
    "make_data_plane",
]


def make_data_plane(worker: "LayerwiseKVPoolWorker") -> LayerwiseDataPlane:
    """Bind the data-plane strategy matching the selected backend."""
    plane = worker.layerwise_data_plane
    if plane == "block_key":
        return BlockKeyDataPlane(worker)
    if plane == "gva":
        return GvaDataPlane(worker)
    raise ValueError(f"Layerwise data plane {plane!r} is not supported by the layerwise worker")
