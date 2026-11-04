# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Allocate pinned private HiSparse storage before Core creates cache views.

Core registers an aligned pageable allocation through ``pin_tensor``. AIV
mapping alone does not make that storage a valid asynchronous DMA destination
on Ascend. Reuse PTA's pinned allocator and Core's existing release ownership.
Remove this adapter when Core exposes a backend Host-allocation hook.
"""

from functools import wraps


def install() -> None:
    from vllm.v1.hisparse import runtime
    from vllm.v1.hisparse.backend import get_hisparse_backend

    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    original = runtime.allocate_pinned_host_pool
    if getattr(original, "_ascend_pinned_private_pool", False):
        return

    @wraps(original)
    def allocate(size):
        backend = get_hisparse_backend()
        if isinstance(backend, AscendHiSparseBackend):
            return backend.allocate_private_host_pool(size)
        return original(size)

    allocate._ascend_pinned_private_pool = True
    runtime.allocate_pinned_host_pool = allocate
