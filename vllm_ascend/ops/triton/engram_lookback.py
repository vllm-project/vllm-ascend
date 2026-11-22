# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/models/deepseek_v41/nvidia/model_state.py
"""Engram lookback kernels; model code owns buffers and launch scheduling."""

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_DEPTH)
    # The mapping contains only the active requests. The window also has padding rows.
    if batch_idx < num_reqs:
        req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        pos = num_computed - 1 - offs
        valid = (offs < DEPTH) & (pos >= 0)
        # Masked lanes on Ascend also need an address inside the request's history row.
        safe_pos = tl.maximum(pos, 0)
        ids = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + safe_pos,
            mask=valid,
            other=-1,
        )
    else:
        ids = tl.full((BLOCK_DEPTH,), -1, all_token_ids_ptr.dtype.element_ty)

    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)
