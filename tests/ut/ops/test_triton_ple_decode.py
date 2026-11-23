# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU contract tests for the Ascend PLE decode launcher."""

import pytest
import torch

from vllm_ascend.ops import triton_ple


class _CaptureKernel:
    def __init__(self):
        self.grid = None
        self.args = None
        self.kwargs = None

    def __getitem__(self, grid):
        self.grid = grid
        return self.launch

    def launch(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "dtype,index_dtype,window_contiguous,outer_enabled,init_enabled,map_enabled,metadata_enabled",
    [
        (torch.bfloat16, torch.int32, False, False, False, False, False),
        (torch.float16, torch.int64, True, True, True, True, True),
        (torch.bfloat16, torch.int64, True, True, False, False, False),
        (torch.float16, torch.int32, False, False, True, False, True),
        (torch.bfloat16, torch.int32, True, False, False, True, False),
        (torch.float16, torch.int64, False, True, False, True, True),
    ],
)
def test_decode_launch_contract(
    monkeypatch, dtype, index_dtype, window_contiguous, outer_enabled, init_enabled, map_enabled, metadata_enabled
):
    capture = _CaptureKernel()
    monkeypatch.setattr(triton_ple, "_ple_conv_decode_kernel", capture)

    tokens, channels, kernel_size, dilation = 3, 7, 4, 3
    inputs = torch.randn(tokens, channels, dtype=dtype)
    residual = torch.randn_like(inputs)
    if window_contiguous:
        state = torch.randn(tokens + 1, channels, 9, dtype=dtype).transpose(1, 2)
    else:
        state = torch.randn(tokens + 1, 9, channels, dtype=dtype)
    weights = torch.randn(channels, kernel_size, dtype=dtype)
    indices = torch.tensor([[1, 0], [2, 0], [3, 0]], dtype=index_dtype)[:, 0]
    outer = torch.randn_like(inputs) if outer_enabled else None
    initialized = torch.tensor([True, False, True]) if init_enabled else None
    token_map = torch.tensor([2, 0, 1], dtype=index_dtype) if map_enabled else None
    query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    accepted = torch.tensor([1, 1, 1], dtype=torch.int32)
    snapshots = [tensor.clone() for tensor in (inputs, residual, state, weights, indices, query_start_loc, accepted)]

    triton_ple.ple_conv_decode(
        inputs,
        residual,
        state,
        weights,
        indices,
        outer,
        dilation=dilation,
        query_start_loc=query_start_loc if metadata_enabled else None,
        num_accepted_tokens=accepted if metadata_enabled else None,
        has_initial_states=initialized,
        spec_query_len=4,
        token_indices=token_map,
    )

    assert capture.grid == (tokens, 1)
    assert capture.args is not None
    assert capture.kwargs is not None
    assert capture.args[0] is inputs
    assert capture.args[1] is state
    assert capture.args[3] is residual
    assert capture.args[4] is outer
    assert capture.args[5] is indices
    assert capture.args[6] is (initialized if init_enabled else indices)
    assert capture.args[7] is (token_map if map_enabled else indices)
    assert capture.args[8] is map_enabled
    assert capture.args[9] == 2  # Non-contiguous state-index vector.
    assert capture.args[10:13] == (state.stride(0), state.stride(1), state.stride(2))
    assert capture.kwargs["HAS_INIT"] is init_enabled
    assert capture.kwargs["FUSE_RESIDUAL"] is outer_enabled
    assert capture.kwargs["DILATION"] == dilation
    assert capture.kwargs["STATE_LEN"] == 9
    assert capture.kwargs["KERNEL_SIZE"] == kernel_size
    assert capture.kwargs["num_warps"] == triton_ple.PLE_NUM_WARPS
    tap_major_weights = capture.args[2]
    assert tap_major_weights.shape == (kernel_size, channels)
    assert tap_major_weights.is_contiguous()
    assert tap_major_weights.data_ptr() != weights.data_ptr()
    torch.testing.assert_close(tap_major_weights, weights.transpose(0, 1))
    for tensor, snapshot in zip((inputs, residual, state, weights, indices, query_start_loc, accepted), snapshots):
        torch.testing.assert_close(tensor, snapshot)


@pytest.mark.parametrize("bad_channels,bad_window", [(True, False), (False, True)])
def test_decode_rejects_invalid_state_shape(bad_channels, bad_window):
    channels = 7
    state = torch.empty(2, 8 if bad_window else 9, channels - int(bad_channels))
    with pytest.raises(ValueError, match="conv_state must have shape"):
        triton_ple.ple_conv_decode(
            torch.empty(1, channels),
            torch.empty(1, channels),
            state,
            torch.empty(channels, 4),
            torch.tensor([1], dtype=torch.int32),
            None,
            dilation=3,
        )
