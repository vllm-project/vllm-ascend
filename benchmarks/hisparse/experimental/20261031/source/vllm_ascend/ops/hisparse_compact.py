# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable sparse-index compaction selected by an Ascend950 eager caller."""

import torch
from vllm_ascend.ops.hisparse_layout import _MAX_TOPK, compact_physical_indices


def compact_physical_indices_eager(indices, capacity, positions):
    """A5-only caller-qualified preview; preserve the production public contract."""
    if indices.ndim != 2 or indices.dtype != torch.int32 or indices.stride(1) != 1:
        raise ValueError("HiSparse physical indices must be int32 [queries, top_k] with contiguous columns")
    topk = indices.shape[1]
    if topk <= 0 or topk > _MAX_TOPK or topk % 128:
        raise ValueError("HiSparse direct attention requires top_k in [128, 32768], divisible by 128")
    if positions.shape != (1, topk) or positions.dtype != torch.float32 or positions.device != indices.device:
        raise ValueError("HiSparse position keys must be FP32 [1, top_k] on the indices device")
    if topk > 2048 or indices.device.type != "npu":
        return compact_physical_indices(indices, capacity, positions)
    from vllm_ascend.ops.triton.hisparse_compact import stable_compact_kernel, triton

    rows = indices.shape[0]
    selected = torch.empty((rows, topk), dtype=torch.int32, device=indices.device)
    counts = torch.empty((rows,), dtype=torch.int32, device=indices.device)
    if rows:
        stable_compact_kernel[(rows,)](
            indices,
            selected,
            counts,
            ROW_STRIDE=indices.stride(0),
            TOP_K=topk,
            CAPACITY=capacity,
            BLOCK=triton.next_power_of_2(topk),
            multibuffer=False,
        )
    return selected, counts
