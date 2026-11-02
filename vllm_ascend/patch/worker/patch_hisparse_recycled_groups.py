# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Candidate Ascend hook for grouped recycled-block invalidation.

Keep upstream's worker unchanged. Remove this version-pinned wrapper once
HiSparse exposes a backend hook for recycled-block invalidation. Unsupported
bindings and graph capture keep the original per-runtime implementation.
"""

import hashlib
import inspect
from functools import wraps
from pathlib import Path

_COMMON_WORKER_SHA256 = "1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229"


def install():
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    original = HiSparseConnectorWorker.invalidate_blocks
    if getattr(original, "_ascend_recycled_groups", False):
        return
    if (
        hashlib.sha256(Path(inspect.getsourcefile(original)).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        != _COMMON_WORKER_SHA256
    ):
        raise RuntimeError("Unreviewed HiSparse recycled-block call site")

    @wraps(original)
    def invalidate(worker, block_ids, request_state_indices):
        if not block_ids:
            return
        handles = worker.cache_handles
        if handles and handles[0].runtime.device.type == "npu":
            from vllm_ascend.ops.triton.hisparse_recycled_invalidate import RecycledGroupInvalidator

            owner = getattr(worker, "_ascend_recycled_group_invalidator", None)
            if owner is None:
                owner = RecycledGroupInvalidator()
                worker._ascend_recycled_group_invalidator = owner
            if owner.try_call(worker.leader_runtimes, block_ids, request_state_indices, worker.kernel_block_size):
                return
        return original(worker, block_ids, request_state_indices)

    invalidate._ascend_recycled_groups = True
    invalidate._ascend_original = original
    HiSparseConnectorWorker.invalidate_blocks = invalidate
