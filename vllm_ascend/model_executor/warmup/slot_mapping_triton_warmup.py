# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the fused multi-group slot-mapping kernels (``ops/triton/compute_slot_mapping.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import HAS_TRITON

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


@torch.inference_mode()
def slot_mapping_triton_warmup(worker: NPUWorker) -> None:
    """JIT every fused slot-mapping specialization before serving.

    Must run after ``initialize_kv_cache``: the fused path only exists on the
    multi-group block table built from the KV-cache config, while the input
    batch seen by ``profile_run`` still has a single group.
    """
    if not HAS_TRITON:
        return
    input_batch = getattr(worker.model_runner, "input_batch", None)
    block_table = getattr(input_batch, "block_table", None)
    prewarm = getattr(block_table, "prewarm_fused_slot_mapping_kernels", None)
    if callable(prewarm):
        prewarm()
