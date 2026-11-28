# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU contract tests for the Ascend PLE speculative (MTP) conv launcher."""

from typing import Any

import pytest
import torch

from vllm_ascend.ops import triton_ple


class _CaptureKernel:
    def __init__(self):
        self.grid: Any = None
        self.args: Any = None
        self.kwargs: Any = None

    def __getitem__(self, grid):
        self.grid = grid
        return self.launch

    def launch(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "dtype,index_dtype,window_contiguous,outer_enabled,map_enabled",
    [
        (torch.bfloat16, torch.int32, False, False, False),
        (torch.float16, torch.int64, True, True, True),
        (torch.bfloat16, torch.int64, True, False, False),
        (torch.float16, torch.int32, False, True, True),
    ],
)
def test_spec_launch_contract(monkeypatch, dtype, index_dtype, window_contiguous, outer_enabled, map_enabled):
    capture_out = _CaptureKernel()
    capture_writeback = _CaptureKernel()
    monkeypatch.setattr(triton_ple, "_ple_conv_spec_kernel", capture_out)
    monkeypatch.setattr(triton_ple, "_ple_conv_spec_writeback_kernel", capture_writeback)

    channels, kernel_size, dilation, spec_query_len = 7, 4, 3, 3
    num_reqs = 2
    tokens = num_reqs * spec_query_len
    state_len = (kernel_size - 1) * dilation
    state_width = state_len + spec_query_len - 1
    inputs = torch.randn(tokens, channels, dtype=dtype)
    residual = torch.randn_like(inputs)
    if window_contiguous:
        state = torch.randn(num_reqs + 1, channels, state_width, dtype=dtype).transpose(1, 2)
    else:
        state = torch.randn(num_reqs + 1, state_width, channels, dtype=dtype)
    weights = torch.randn(channels, kernel_size, dtype=dtype)
    indices = torch.tensor([[1, 0], [2, 0]], dtype=index_dtype)[:, 0]
    outer = torch.randn_like(inputs) if outer_enabled else None
    token_map = torch.arange(tokens - 1, -1, -1, dtype=index_dtype) if map_enabled else None
    query_start_loc = torch.tensor([0, spec_query_len, tokens], dtype=torch.int32)
    accepted = torch.tensor([1, spec_query_len], dtype=torch.int32)
    snapshots = [tensor.clone() for tensor in (inputs, residual, state, weights, indices, query_start_loc, accepted)]

    triton_ple.ple_conv_spec(
        inputs,
        residual,
        state,
        weights,
        indices,
        outer,
        dilation=dilation,
        query_start_loc=query_start_loc,
        num_accepted_tokens=accepted,
        spec_query_len=spec_query_len,
        token_indices=token_map,
    )

    block_c = triton_ple.PLE_BLOCK_C
    assert capture_out.grid == (tokens, triton_ple.triton.cdiv(channels, block_c))
    assert capture_writeback.grid == (num_reqs, triton_ple.triton.cdiv(channels, block_c))

    out_args = capture_out.args
    assert out_args[0] is inputs
    assert out_args[1] is state
    assert out_args[3] is residual
    assert out_args[4] is outer
    assert out_args[5] is indices
    assert out_args[6] is query_start_loc
    assert out_args[7] is accepted
    assert out_args[8] is (token_map if map_enabled else indices)
    assert out_args[9] is map_enabled
    assert out_args[10] == num_reqs
    assert out_args[11] == max(num_reqs, 1).bit_length()
    assert out_args[12] == 2  # Non-contiguous state-index vector.
    assert out_args[13:16] == state.stride()
    assert capture_out.kwargs["C"] == channels
    assert capture_out.kwargs["BLOCK_C"] == block_c
    assert capture_out.kwargs["STATE_LEN"] == state_len
    assert capture_out.kwargs["DILATION"] == dilation
    assert capture_out.kwargs["KERNEL_SIZE"] == kernel_size
    assert capture_out.kwargs["SPEC_QUERY_LEN"] == spec_query_len
    assert capture_out.kwargs["FUSE_RESIDUAL"] is outer_enabled
    assert capture_out.kwargs["num_warps"] == triton_ple.PLE_NUM_WARPS

    wb_args = capture_writeback.args
    assert wb_args[0] is inputs
    assert wb_args[1] is state
    assert wb_args[2] is indices
    assert wb_args[3] is query_start_loc
    assert wb_args[4] is accepted
    assert wb_args[5] is (token_map if map_enabled else indices)
    assert wb_args[6] is map_enabled
    assert wb_args[7] == 2
    assert wb_args[8:11] == state.stride()
    assert capture_writeback.kwargs["STATE_LEN"] == state_len
    assert capture_writeback.kwargs["SPEC_QUERY_LEN"] == spec_query_len
    assert capture_writeback.kwargs["STATE_WIDTH"] == state_width
    assert capture_writeback.kwargs["num_warps"] == triton_ple.PLE_NUM_WARPS

    tap_major_weights = out_args[2]
    assert tap_major_weights.shape == (kernel_size, channels)
    assert tap_major_weights.is_contiguous()
    assert tap_major_weights.data_ptr() != weights.data_ptr()
    torch.testing.assert_close(tap_major_weights, weights.transpose(0, 1))
    for tensor, snapshot in zip((inputs, residual, state, weights, indices, query_start_loc, accepted), snapshots):
        torch.testing.assert_close(tensor, snapshot)


@pytest.mark.parametrize("missing", ["query_start_loc", "num_accepted_tokens"])
def test_spec_requires_spec_metadata(missing):
    channels, tokens = 7, 3
    kwargs = dict(
        dilation=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
    )
    kwargs[missing] = None
    with pytest.raises(ValueError, match="query_start_loc and num_accepted_tokens"):
        triton_ple.ple_conv_spec(
            torch.empty(tokens, channels),
            torch.empty(tokens, channels),
            torch.empty(2, 9, channels),
            torch.empty(channels, 4),
            torch.tensor([1], dtype=torch.int32),
            None,
            **kwargs,
        )


@pytest.mark.parametrize("bad_channels,bad_window", [(True, False), (False, True)])
def test_spec_rejects_invalid_state_shape(bad_channels, bad_window):
    channels, spec_query_len = 7, 3
    state_width = 9 + spec_query_len - 1
    state = torch.empty(2, state_width - int(bad_window), channels - int(bad_channels))
    with pytest.raises(ValueError, match="conv_state must have shape"):
        triton_ple.ple_conv_spec(
            torch.empty(1, channels),
            torch.empty(1, channels),
            state,
            torch.empty(channels, 4),
            torch.tensor([1], dtype=torch.int32),
            None,
            dilation=3,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
            spec_query_len=spec_query_len,
        )
