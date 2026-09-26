# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendC short convolution for GLM prefill, decode and MTP verification."""

import torch
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

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
    """Consume GDN metadata and update the caller's [cache, state_len, dim] state."""
    # Padded requests can be skipped by the kernel; their output must stay zero.
    output = torch.zeros_like(x)
    if cache_indices.shape[0] == 0:
        return output
    kernel_state = conv_state
    kernel_indices = cache_indices
    # aclnnCausalConv1d materializes a non-contiguous state without writing its
    # mutations back to the view. Stage only this batch's rows, retaining both
    # page strides and DS layouts; never copy the entire persistent cache.
    if not conv_state.is_contiguous():
        requests = cache_indices.shape[0]
        state_len, dim = conv_state.shape[1:]
        kernel_state = torch.empty((requests, state_len, dim), dtype=conv_state.dtype, device=conv_state.device)
        kernel_indices = torch.empty(requests, dtype=torch.int32, device=cache_indices.device)
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=False)
    kernel_indices = kernel_indices.contiguous()
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
            null_block_id=0,
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
            null_block_id=0,
        )
    if not conv_state.is_contiguous():
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=True)
    return result
