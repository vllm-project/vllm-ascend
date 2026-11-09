# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GVA plane: the pool stores data behind worker-allocated virtual addresses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes.base import (
    LayerwiseDataPlane,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import (
        LayerwiseKVPoolWorker,
    )


class GvaDataPlane(LayerwiseDataPlane):
    """Route the GVA plane's step entry points to the worker implementation.

    The GVA protocol derives its transfer targets from per-step worker state
    (block tables, partial-block snapshots, pool GVA metadata), so target
    preparation stays on the worker; the plane exists so a step drives both
    backend planes through one interface. Pool I/O flows through the
    worker's helpers unchanged.
    """

    def __init__(self, worker: LayerwiseKVPoolWorker):
        self._worker = worker

    def prepare_sessions(self, requests: list[ReqMeta]) -> dict[int, list[ReqMeta]]:
        """GVA opens no pool-side sessions before the save loop."""
        return {}

    def prepare_load_gvas(self, requests: list[ReqMeta]) -> None:
        self._worker._prepare_load_gvas(requests)

    def alloc_save_gvas(self, requests: list[ReqMeta]) -> None:
        self._worker._alloc_gvas_for_save(requests)

    def finish_current_load_sessions(self) -> None:
        """GVA loads read through worker-held addresses, not pool sessions."""
