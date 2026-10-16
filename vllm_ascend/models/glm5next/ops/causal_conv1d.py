# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendC short convolution for GLM prefill, decode and MTP verification."""

import torch
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    run_mode: int,
    initial_state_mode: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update GDN state in place; cache indices use slot zero for padding."""
    if cache_indices.shape[0] == 0:
        return torch.zeros_like(x)
    kernel_state = conv_state
    # FLA requires 1D indices. ACLNN still copies a strided column internally.
    kernel_indices = cache_indices[:, 0] if cache_indices.ndim == 2 else cache_indices
    null_block_id = NULL_BLOCK_ID
    # AscendC reads and updates page-strided SD states in place. Only stage
    # layouts outside its stride contract, such as a transposed DS state whose
    # channel dimension is not contiguous. Copy only this batch's state rows.
    needs_state_copy = conv_state.stride(0) <= 0 or conv_state.stride(1) <= 0 or conv_state.stride(2) != 1
    if needs_state_copy:
        requests = cache_indices.shape[0]
        state_len, dim = conv_state.shape[1:]
        kernel_state = torch.empty((requests, state_len, dim), dtype=conv_state.dtype, device=conv_state.device)
        kernel_indices = torch.empty(requests, dtype=torch.int32, device=cache_indices.device)
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=False)
        # Packed row zero is live; the gather kernel uses requests for padding.
        null_block_id = requests
    if run_mode == 0:
        result = causal_conv1d_fn(
            x,
            weight,
            None,
            conv_states=kernel_state,
            query_start_loc=query_start_loc,
            cache_indices=kernel_indices,
            has_initial_state=initial_state_mode,
            activation="silu",
            pad_slot_id=PAD_SLOT_ID,
            null_block_id=null_block_id,
        )
    else:
        result = causal_conv1d_update(
            x,
            kernel_state,
            weight,
            bias=None,
            activation="silu",
            conv_state_indices=kernel_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            null_block_id=null_block_id,
        )
    if needs_state_copy:
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=True)
    return result
