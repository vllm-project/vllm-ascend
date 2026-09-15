# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendC short convolution for GLM prefill, decode and MTP verification."""

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

CONV_STATE_COPY_BLOCK_SIZE = 256


@triton.jit
def _copy_conv_state(
    cache,
    packed,
    cache_indices,
    starts,
    packed_indices,
    cache_stride,
    index_stride,
    num_slots,
    STATE_LEN: tl.constexpr,
    DIM: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    DIM_STRIDE: tl.constexpr,
    WRITE_BACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(cache_indices + request * index_stride).to(tl.int64)
    active = (slot >= 0) & (slot < num_slots) & (tl.load(starts + request + 1) > tl.load(starts + request))
    in_range = offsets < STATE_LEN * DIM
    safe_slot = tl.where(active, slot, 0)
    cache_offsets = safe_slot * cache_stride + offsets // DIM * STATE_STRIDE + offsets % DIM * DIM_STRIDE
    packed_offsets = request * STATE_LEN * DIM + offsets
    if WRITE_BACK:
        values = tl.load(packed + packed_offsets, mask=in_range, other=0)
        tl.store(cache + cache_offsets, values, mask=active & in_range)
    else:
        values = tl.load(cache + cache_offsets, mask=active & in_range, other=0)
        tl.store(packed + packed_offsets, values, mask=in_range)
        if tl.program_id(1) == 0:
            tl.store(packed_indices + request, tl.where(active, request, -1))


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
        copy_grid = (requests, triton.cdiv(state_len * dim, CONV_STATE_COPY_BLOCK_SIZE))
        copy_args = (
            conv_state,
            kernel_state,
            cache_indices,
            query_start_loc,
            kernel_indices,
            conv_state.stride(0),
            cache_indices.stride(0),
            conv_state.shape[0],
            state_len,
            dim,
            conv_state.stride(1),
            conv_state.stride(2),
        )
        _copy_conv_state[copy_grid](*copy_args, WRITE_BACK=False, BLOCK=CONV_STATE_COPY_BLOCK_SIZE)
    # Return the declared result so graph functionalization retains the call.
    result = torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        weight,
        conv_state=kernel_state,
        bias_opt=None,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=kernel_indices,
        initial_state_mode_opt=initial_state_mode,
        num_accepted_tokens_opt=num_accepted_tokens,
        activation_mode=1,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=run_mode,
    )
    if not conv_state.is_contiguous():
        _copy_conv_state[copy_grid](*copy_args, WRITE_BACK=True, BLOCK=CONV_STATE_COPY_BLOCK_SIZE)
    return result
