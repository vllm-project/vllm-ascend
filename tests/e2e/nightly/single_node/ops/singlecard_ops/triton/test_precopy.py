# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.mamba.precopy import precopy_mamba_align_fused_kernel
from vllm_ascend.ops.triton.v1.mamba.postprocess import postprocess_mamba_fused_kernel

NUM_LAYERS = 2
CONV_WIDTH = 4
CONV_DIM = 64
SSM_SHAPE = (4, 16, 16)
MAX_COLS = 8


def _build_states(num_blocks, device, conv_dim_first):
    conv_shape = (num_blocks, CONV_DIM, CONV_WIDTH) if conv_dim_first else (num_blocks, CONV_WIDTH, CONV_DIM)
    convs = [torch.randn(conv_shape, dtype=torch.bfloat16, device=device) for _ in range(NUM_LAYERS)]
    ssms = [torch.randn((num_blocks, *SSM_SHAPE), dtype=torch.float32, device=device) for _ in range(NUM_LAYERS)]
    return convs, ssms


def _build_meta(convs, ssms, device, conv_dim_first):
    base: list[int] = []
    block_stride: list[int] = []
    elem_size: list[int] = []
    inner_size: list[int] = []
    conv_width: list[int] = []
    row_count: list[int] = []
    row_stride: list[int] = []
    for conv, ssm in zip(convs, ssms):
        base.extend((conv.data_ptr(), ssm.data_ptr()))
        block_stride.extend((conv.stride(0) * conv.element_size(), ssm.stride(0) * ssm.element_size()))
        elem_size.extend((conv.element_size(), ssm.element_size()))
        inner_size.extend((1 if conv_dim_first else conv.stride(1), ssm[0].numel()))
        conv_width.extend((CONV_WIDTH, 0))
        row_count.extend((CONV_DIM if conv_dim_first else 0, 0))
        row_stride.extend((conv.stride(1) * conv.element_size() if conv_dim_first else 0, 0))
    tensor = lambda values, dtype: torch.tensor(values, dtype=dtype, device=device)
    state_count = NUM_LAYERS * 2
    return (
        tensor(base, torch.int64),
        tensor(block_stride, torch.int64),
        tensor(elem_size, torch.int32),
        tensor(inner_size, torch.int64),
        tensor(conv_width, torch.int32),
        torch.zeros(state_count, dtype=torch.int32, device=device),
        tensor(row_count, torch.int32),
        tensor(row_stride, torch.int64),
    )


def _reference(convs, ssms, block_table, src_col, dst_col, bias, conv_dim_first, slots=None):
    conv_before, ssm_before = [state.clone() for state in convs], [state.clone() for state in ssms]
    conv_ref, ssm_ref = [state.clone() for state in convs], [state.clone() for state in ssms]
    for req_idx in range(len(src_col)) if slots is None else slots:
        if req_idx < 0:
            continue
        src, dst, token_bias = int(src_col[req_idx]), int(dst_col[req_idx]), int(bias[req_idx])
        if src < 0 or src == dst:
            continue
        src_block, dst_block = int(block_table[req_idx, src]), int(block_table[req_idx, dst])
        temporal_block = int(block_table[req_idx, src + token_bias])
        for layer in range(NUM_LAYERS):
            if conv_dim_first:
                conv_ref[layer][dst_block, :, : CONV_WIDTH - token_bias] = conv_before[layer][src_block, :, token_bias:]
            else:
                conv_ref[layer][dst_block, : CONV_WIDTH - token_bias] = conv_before[layer][src_block, token_bias:]
            ssm_ref[layer][dst_block] = ssm_before[layer][temporal_block]
    return conv_ref, ssm_ref


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("conv_dim_first", [False, True])
@pytest.mark.parametrize("slots", [None, [0, 1, 2, 3], [3, 2, -1]])
@pytest.mark.parametrize("temporal_tiles", [1, 4])
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_precopy_matches_reference(conv_dim_first, slots, temporal_tiles):
    device = torch.device("npu:0")
    torch.manual_seed(0)
    num_reqs = 4
    num_blocks = num_reqs * MAX_COLS + 1
    block_table = torch.arange(1, num_blocks, dtype=torch.int32, device=device).reshape(num_reqs, MAX_COLS)
    src_col = torch.tensor([-1, 1, 1, 1], dtype=torch.int32, device=device)
    dst_col = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device=device)
    bias = torch.tensor([0, 0, 1, 2], dtype=torch.int32, device=device)
    convs, ssms = _build_states(num_blocks, device, conv_dim_first)
    conv_ref, ssm_ref = _reference(
        convs, ssms, block_table.cpu(), src_col.cpu(), dst_col.cpu(), bias.cpu(), conv_dim_first, slots
    )
    base, block_stride, elem_size, inner_size, conv_width, group, row_count, row_stride = _build_meta(
        convs, ssms, device, conv_dim_first
    )
    block_table_ptrs = torch.tensor([block_table.data_ptr()], dtype=torch.int64, device=device)
    idx_mapping = torch.tensor(slots, dtype=torch.int32, device=device) if slots is not None else None
    num_reqs = len(slots) if slots is not None else num_reqs
    grid = (num_reqs, NUM_LAYERS * 2, temporal_tiles)
    precopy_mamba_align_fused_kernel[grid](
        dst_col,
        src_col,
        bias,
        block_table_ptrs,
        block_table.stride(0),
        base,
        block_stride,
        elem_size,
        inner_size,
        conv_width,
        group,
        row_count,
        row_stride,
        idx_mapping,
        num_reqs,
        COPY_BLOCK_SIZE=128,
        CONV_STATE_DIM_FIRST=conv_dim_first,
        HAS_IDX_MAPPING=slots is not None,
        TEMPORAL_TILES=temporal_tiles,
    )
    torch.accelerator.synchronize()
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(convs[layer], conv_ref[layer], rtol=0, atol=0)
        torch.testing.assert_close(ssms[layer], ssm_ref[layer], rtol=0, atol=0)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("conv_dim_first", [False, True])
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_postprocess_uses_request_slot_tables(conv_dim_first):
    device = torch.device("npu:0")
    num_blocks = 4 * MAX_COLS + 1
    table = torch.arange(1, num_blocks, dtype=torch.int32, device=device).reshape(4, MAX_COLS)
    convs, ssms = _build_states(num_blocks, device, conv_dim_first)
    # Batch rows are reordered and sparse; slot 1 and the -1 row must stay untouched.
    slots = [3, 0, -1]
    columns = torch.tensor([1, 1, 1, 1], dtype=torch.int32, device=device)
    computed = torch.tensor([17, 17, 17, 17], dtype=torch.int32, device=device)
    accepted = torch.tensor([3, 3, 3, 3], dtype=torch.int32, device=device)
    # running=15, aligned=16: copy the accepted candidate at bias 1 to column 0.
    expected_conv, expected_ssm = _reference(
        convs, ssms, table.cpu(), columns.cpu(), [0] * 4, [1] * 4, conv_dim_first, slots
    )
    metadata = _build_meta(convs, ssms, device, conv_dim_first)
    postprocess_mamba_fused_kernel[(len(slots), NUM_LAYERS * 2, 1)](
        accepted.clone(),
        columns,
        None,
        computed,
        None,
        torch.tensor([table.data_ptr()], dtype=torch.int64, device=device),
        table.stride(0),
        *metadata,
        accepted,
        torch.tensor(slots, dtype=torch.int32, device=device),
        len(slots),
        block_size=16,
        COPY_BLOCK_SIZE=128,
        CONV_STATE_DIM_FIRST=conv_dim_first,
        HAS_IDX_MAPPING=True,
        PRECOMPUTED_NEW_COMPUTED=True,
    )
    torch.accelerator.synchronize()
    for actual, expected in zip([*convs, *ssms], [*expected_conv, *expected_ssm]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert accepted.tolist() == [3, 3, 3, 3]
