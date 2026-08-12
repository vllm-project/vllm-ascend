# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import random

import numpy as np
import pytest
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.mamba.postprocess import _copy_mamba_state_block

seed = 45
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

_NUM_BLOCKS = 8
# Mirrors the production tuning in MambaSpecDecodeGPUContext.run_fused_postprocess.
_COPY_BLOCK_SIZE = 1024


@triton.jit
def _launch_copy_kernel(
    scalars_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req: tl.int64,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
):
    # The index scalars must reach _copy_mamba_state_block as runtime triton
    # scalars (it calls ``token_bias.to(tl.int64)`` etc.), exactly like the
    # fused kernels do. Loading them from a tensor avoids python-int
    # specialization, which would break the ``.to(...)`` casts.
    state_idx = tl.load(scalars_ptr + 0)
    bt_row_idx = tl.load(scalars_ptr + 1)
    src_col = tl.load(scalars_ptr + 2)
    dst_col = tl.load(scalars_ptr + 3)
    token_bias = tl.load(scalars_ptr + 4)
    _copy_mamba_state_block(
        state_idx,
        bt_row_idx,
        src_col,
        dst_col,
        token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
        state_base_addrs_ptr,
        state_block_strides_ptr,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_group_indices_ptr,
        state_dim_row_count_ptr,
        state_dim_row_stride_ptr,
        COPY_BLOCK_SIZE,
        CONV_STATE_DIM_FIRST,
    )


def _layout_spec(layout, conv_width, dim, temporal_dim):
    """Return the state shape and per-state metadata for a layout.

    Mirrors the metadata MambaSpecDecodeGPUContext.initialize_from_forward_context
    derives from each state tensor:
    - temporal state [num_blocks, temporal_dim]: conv_width 0, full-block copy
    - SD conv state  [num_blocks, conv_width, dim]: dim is the inner axis
    - DS conv state  [num_blocks, dim, conv_width]: state_len is the slide axis
    """
    if layout == "temporal":
        shape = (_NUM_BLOCKS, temporal_dim)
        return shape, 0, temporal_dim, False, 0, 0, temporal_dim
    if layout == "sd_conv":
        shape = (_NUM_BLOCKS, conv_width, dim)
        return shape, conv_width, dim, False, 0, 0, conv_width * dim
    if layout == "ds_conv":
        shape = (_NUM_BLOCKS, dim, conv_width)
        return shape, conv_width, 1, True, dim, conv_width, dim * conv_width
    raise ValueError(f"unknown layout {layout}")


def _golden_copy(state, block_table, src_col, dst_col, token_bias, conv_width,
                 dim_first):
    """Pure-torch reference for _copy_mamba_state_block (single state slot)."""
    out = state.clone()
    bt_row = block_table[0]
    dst_block = int(bt_row[dst_col].item())
    if conv_width > 0 and dim_first:
        # DS conv: state[src, :, token_bias:] -> state[dst, :, :conv_width-bias]
        src_block = int(bt_row[src_col].item())
        n = conv_width - token_bias
        out[dst_block, :, 0:n] = state[src_block, :, token_bias:conv_width]
    elif conv_width > 0:
        # SD conv: state[src, token_bias:, :] -> state[dst, :conv_width-bias, :]
        src_block = int(bt_row[src_col].item())
        n = conv_width - token_bias
        out[dst_block, 0:n, :] = state[src_block, token_bias:conv_width, :]
    else:
        # Temporal: state[bt[src_col + token_bias]] -> state[dst]
        src_block = int(bt_row[src_col + token_bias].item())
        out[dst_block] = state[src_block]
    return out


def _build_metadata(state, block_table, conv_width_meta, inner_size, dim_first,
                    dim_rows, row_stride_elems, block_stride_elems, device):
    """Build the per-state metadata tensors the device function reads.

    Dtypes match MambaSpecDecodeGPUContext.create exactly: int64 for addresses
    and byte strides, int32 for element/conv-width/group/row-count, int64 for
    inner sizes and row strides.
    """
    elem_size = state.element_size()
    state_base_addrs = torch.tensor([state.data_ptr()], dtype=torch.int64, device=device)
    state_block_strides = torch.tensor(
        [block_stride_elems * elem_size], dtype=torch.int64, device=device)
    state_elem_sizes = torch.tensor([elem_size], dtype=torch.int32, device=device)
    state_inner_sizes = torch.tensor([inner_size], dtype=torch.int64, device=device)
    state_conv_widths = torch.tensor([conv_width_meta], dtype=torch.int32, device=device)
    state_group_indices = torch.tensor([0], dtype=torch.int32, device=device)
    state_dim_row_count = torch.tensor([dim_rows], dtype=torch.int32, device=device)
    state_dim_row_stride = torch.tensor(
        [row_stride_elems * elem_size], dtype=torch.int64, device=device)
    block_table_ptrs = torch.tensor(
        [block_table.data_ptr()], dtype=torch.int64, device=device)
    block_table_stride_req = int(block_table.stride(0))
    return (
        state_base_addrs, state_block_strides, state_elem_sizes,
        state_inner_sizes, state_conv_widths, state_group_indices,
        state_dim_row_count, state_dim_row_stride, block_table_ptrs,
        block_table_stride_req,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("token_bias", [0, 1, 2])
@pytest.mark.parametrize("size", ["small", "large"])
@pytest.mark.parametrize("layout", ["temporal", "sd_conv", "ds_conv"])
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_copy_mamba_state_block(layout, size, token_bias, dtype):
    sizes = {
        "small": {"conv_width": 4, "dim": 32, "temporal_dim": 64},
        "large": {"conv_width": 8, "dim": 64, "temporal_dim": 512},
    }
    conv_width = sizes[size]["conv_width"]
    dim = sizes[size]["dim"]
    temporal_dim = sizes[size]["temporal_dim"]

    (shape, conv_width_meta, inner_size, dim_first, dim_rows, row_stride_elems,
     block_stride_elems) = _layout_spec(layout, conv_width, dim, temporal_dim)

    # token_bias must stay inside the conv window for conv states, and
    # src_col + token_bias must be a valid block column for temporal states.
    assert token_bias < conv_width or conv_width_meta == 0
    src_col = 0
    dst_col = _NUM_BLOCKS - 1
    assert src_col + token_bias < _NUM_BLOCKS
    # Keep src and dst blocks distinct so the copy is observable.
    assert src_col + token_bias != dst_col

    device = torch.device("npu:0")
    state = torch.randn(*shape, dtype=dtype, device=device)
    state_ref = state.cpu().clone()

    block_table = torch.arange(_NUM_BLOCKS, dtype=torch.int32, device=device).view(
        1, _NUM_BLOCKS)

    (state_base_addrs, state_block_strides, state_elem_sizes, state_inner_sizes,
     state_conv_widths, state_group_indices, state_dim_row_count,
     state_dim_row_stride, block_table_ptrs,
     block_table_stride_req) = _build_metadata(
        state, block_table, conv_width_meta, inner_size, dim_first, dim_rows,
        row_stride_elems, block_stride_elems, device)

    # Launch the device function through a single-program wrapper that loads
    # the index scalars as runtime triton scalars, mirroring how the fused
    # kernels feed accept_token_bias / src_col / dst_col into the helper.
    scalars = torch.tensor(
        [0, 0, src_col, dst_col, token_bias], dtype=torch.int64, device=device)
    _launch_copy_kernel[(1,)](
        scalars,
        block_table_ptrs,
        block_table_stride_req,
        state_base_addrs,
        state_block_strides,
        state_elem_sizes,
        state_inner_sizes,
        state_conv_widths,
        state_group_indices,
        state_dim_row_count,
        state_dim_row_stride,
        COPY_BLOCK_SIZE=_COPY_BLOCK_SIZE,
        CONV_STATE_DIM_FIRST=dim_first,
    )
    torch.accelerator.synchronize()

    expected = _golden_copy(state_ref, block_table.cpu(), src_col, dst_col,
                            token_bias, conv_width_meta, dim_first)
    # The copy is a raw byte move, so the result must match bit-for-bit.
    torch.testing.assert_close(state.cpu(), expected, rtol=0, atol=0)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
