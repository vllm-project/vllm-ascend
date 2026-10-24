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
    output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Consume GDN metadata and update the caller's [cache, state_len, dim] state."""
    output = None
    if output_buffer is not None:
        output = output_buffer[: x.shape[0]]
        if output.shape != x.shape:
            raise ValueError("Convolution output buffer must cover the input rows and match its feature shape")
    if cache_indices.shape[0] == 0:
        return torch.zeros_like(x) if output is None else output
    kernel_state = conv_state
    # GLM metadata shares a contiguous vector across this group's layers.
    # Standalone callers can still pass a matrix, using its first column.
    kernel_indices = cache_indices[:, 0] if cache_indices.ndim == 2 else cache_indices
    null_block_id = 0
    # AscendC updates page- and row-strided SD states in place. Keep staging
    # for unsupported layouts, such as DS views with strided channels.
    needs_state_copy = conv_state.stride(0) <= 0 or conv_state.stride(1) <= 0 or conv_state.stride(2) != 1
    if needs_state_copy:
        requests = cache_indices.shape[0]
        state_len, dim = conv_state.shape[1:]
        # UPDATE does not recognize PAD_SLOT_ID: a negative null_block_id
        # disables its null-block check. Reserve a non-negative slot after
        # the request rows, keeping packed request zero a valid state.
        null_block_id = requests
        kernel_state = torch.empty((requests + 1, state_len, dim), dtype=conv_state.dtype, device=conv_state.device)
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
            null_block_id=null_block_id,
        )
    else:
        # Metadata resets the shared destination once per step. Standalone
        # callers retain the original zero-initialized padding behavior.
        if output is None:
            output = torch.zeros_like(x)
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
            out=output,
        )
    if needs_state_copy:
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=True)
    return result
