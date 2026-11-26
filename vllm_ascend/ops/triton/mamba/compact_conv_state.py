# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact accepted RecoverSSM convolution history."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.device.device_config import is_950

RECOVERSSM_CONV_TILE = 4096
RECOVERSSM_CONV_STRIDED_TILE = 256


@triton.jit
def _copy_conv_state_tile(
    source_ptr,
    final_ptr,
    boundary_ptr,
    channel_offsets,
    history_offsets,
    token_stride,
    commit_len,
    boundary_recovery_len,
    mask,
    boundary_valid,
    ALIGN_MODE: tl.constexpr,
):
    final_values = tl.load(
        source_ptr + channel_offsets + (commit_len - 1 + history_offsets) * token_stride,
        mask=mask,
    )
    if ALIGN_MODE:
        boundary_values = tl.load(
            source_ptr + channel_offsets + (boundary_recovery_len - 1 + history_offsets) * token_stride,
            mask=mask & boundary_valid,
        )
        tl.store(
            boundary_ptr + channel_offsets + history_offsets * token_stride,
            boundary_values,
            mask=mask & boundary_valid,
        )
    tl.store(
        final_ptr + channel_offsets + history_offsets * token_stride,
        final_values,
        mask=mask,
    )


@triton.jit
def _compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    conv_dim,
    conv_history_len,
    stride_state_indices,
    BLOCK_D: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    DIM_STRIDE: tl.constexpr = None,
    USE_1D_CHANNEL_GATHER: tl.constexpr = False,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_l = tl.program_id(2)
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_state_idx <= null_block_id:
        return

    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
    block_stride = tl.load(conv_state_block_strides_ptr + pid_l)
    if DIM_STRIDE is None:
        dim_stride = tl.load(conv_state_dim_strides_ptr + pid_l)
    else:
        dim_stride = DIM_STRIDE
    token_stride = tl.load(conv_state_token_strides_ptr + pid_l)
    conv_state_ptr = base_addr.to(tl.pointer_type(conv_state_ref_ptr.dtype.element_ty))
    source_ptr = conv_state_ptr + source_state_idx * block_stride
    final_ptr = conv_state_ptr + final_state_idx * block_stride

    boundary_ptr = None
    if ALIGN_MODE:
        boundary_ptr = conv_state_ptr + boundary_state_idx * block_stride

    if USE_1D_CHANNEL_GATHER and (DIM_STRIDE is not None and DIM_STRIDE != 1):
        # A3 Triton 3.2 requires one-dimensional gathers for strided channels.
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        for history in tl.static_range(BLOCK_HISTORY):
            mask = (offs_d < conv_dim) & (history < conv_history_len)
            _copy_conv_state_tile(
                source_ptr,
                final_ptr,
                boundary_ptr,
                offs_d * dim_stride,
                history,
                token_stride,
                commit_len,
                boundary_recovery_len,
                mask,
                boundary_state_idx > null_block_id,
                ALIGN_MODE,
            )
        return
    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D))[None, :]
    offs_h = tl.arange(0, BLOCK_HISTORY)[:, None]
    mask = (offs_d < conv_dim) & (offs_h < conv_history_len)
    _copy_conv_state_tile(
        source_ptr,
        final_ptr,
        boundary_ptr,
        offs_d * dim_stride,
        offs_h,
        token_stride,
        commit_len,
        boundary_recovery_len,
        mask,
        boundary_state_idx > null_block_id,
        ALIGN_MODE,
    )


def compact_conv_state(
    conv_state_ref: torch.Tensor,
    conv_state_base_addrs: torch.Tensor,
    conv_state_block_strides: torch.Tensor,
    conv_state_dim_strides: torch.Tensor,
    conv_state_token_strides: torch.Tensor,
    state_indices: torch.Tensor,
    commit_lens: torch.Tensor,
    final_state_indices: torch.Tensor,
    boundary_state_indices: torch.Tensor,
    boundary_recovery_lens: torch.Tensor,
    *,
    conv_history_len: int,
    align_mode: bool,
    dim_stride: int | None,
    null_block_id: int,
) -> None:
    """Copy accepted convolution histories into the planned cache pages."""
    batch = state_indices.numel()
    conv_dim = conv_state_ref.shape[1]
    num_layers = conv_state_base_addrs.numel()
    is_a5 = is_950()
    # A3 strided gathers need smaller tiles to fit the UB.
    block_d = RECOVERSSM_CONV_TILE if is_a5 or dim_stride == 1 else RECOVERSSM_CONV_STRIDED_TILE
    grid = (triton.cdiv(conv_dim, block_d), batch, num_layers)
    _compact_conv_state_kernel[grid](
        conv_state_ref,
        conv_state_base_addrs,
        conv_state_block_strides,
        conv_state_dim_strides,
        conv_state_token_strides,
        state_indices,
        commit_lens,
        final_state_indices,
        boundary_state_indices,
        boundary_recovery_lens,
        null_block_id,
        conv_dim,
        conv_history_len,
        state_indices.stride(0),
        BLOCK_D=block_d,
        BLOCK_HISTORY=triton.next_power_of_2(conv_history_len),
        ALIGN_MODE=align_mode,
        DIM_STRIDE=dim_stride,
        USE_1D_CHANNEL_GATHER=not is_a5,
        num_warps=4,
    )
