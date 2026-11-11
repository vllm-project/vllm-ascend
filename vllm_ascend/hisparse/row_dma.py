# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Prepare HiSparse row-mirror DMA through Core's descriptor ownership."""

from functools import wraps

import torch


def _install_worker(worker_type, backend_type, bind_recycled) -> None:
    original_init = worker_type._init_dma
    if getattr(original_init, "_ascend_row_dma", False):
        return

    @wraps(original_init)
    def init(worker):
        original_init(worker)
        if isinstance(worker.backend, backend_type):
            if worker.is_host_writer and worker.shared_host_region is None:
                # Core allocates int64 staging; the bound Ascend slot producer uses int32.
                # Cache binding is complete here and precedes graph capture.
                staging = worker._slot_mapping_staging
                slots = worker.cache_handles[0].slot_mapping
                if staging is not None and not staging.num_tokens and slots.device.type == "npu":
                    if slots.dtype == torch.int32:
                        staging.slots = torch.empty(
                            staging.slots.shape, dtype=slots.dtype, device="cpu", pin_memory=True
                        )
            worker._ascend_recycled_invalidator = bind_recycled(worker)

    init._ascend_row_dma = True
    worker_type._init_dma = init
    original_invalidate = worker_type.invalidate_blocks

    @wraps(original_invalidate)
    def invalidate(worker, block_ids, request_state_indices):
        owner = getattr(worker, "_ascend_recycled_invalidator", None)
        if owner is None or not owner.try_call(worker, block_ids, request_state_indices):
            return original_invalidate(worker, block_ids, request_state_indices)

    worker_type.invalidate_blocks = invalidate


def install(backend_type) -> None:
    """Match Ascend slot dtype and install the recycled-slot adapter."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker

    from vllm_ascend.hisparse.recycled_invalidate import RecycledGroupInvalidator

    _install_worker(worker.HiSparseConnectorWorker, backend_type, RecycledGroupInvalidator.bind)
