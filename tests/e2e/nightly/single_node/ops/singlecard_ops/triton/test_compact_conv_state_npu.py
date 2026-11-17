# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact copy and page routing tests for RecoverSSM convolution compaction."""

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.mamba.compact_conv_state import compact_conv_state  # noqa: E402

HISTORY = 3


@pytest.fixture
def npu():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(419)
    return torch.device("npu")


@pytest.mark.parametrize(
    "channels,channel_first,align,batch,layers",
    [
        (127, True, False, 4, 2),
        (2048, False, False, 11, 11),  # Qwen3.6 TP4 channels; synthetic B11/L11.
        (2560, True, True, 4, 2),  # Qwen3.8 TP4
        (5120, False, True, 4, 2),  # Qwen3.8 TP2
    ],
)
def test_compaction_page_routing(npu, channels, channel_first, align, batch, layers):
    """Check the shared convolution wrapper independently of state recovery."""
    pages, span = 7, 7
    physical = (channels, span) if channel_first else (span, channels)
    backings = [
        torch.full((pages, channels * span + gap), -3, dtype=torch.bfloat16, device=npu)
        for gap in (128 + 128 * layer for layer in range(layers))
    ]
    views, expected = [], []
    for backing in backings:
        golden = backing.clone()
        page_stride = backing.stride(0)
        inner = (physical[1], 1)
        view = backing.as_strided((pages, *physical), (page_stride, *inner))
        view.copy_(torch.randn_like(view))
        golden.copy_(backing)
        target = golden.as_strided(view.shape, view.stride())
        if not channel_first:
            view, target = view.transpose(-1, -2), target.transpose(-1, -2)
        target[3, :, :HISTORY] = view[1, :, 3:6]
        target[2, :, :HISTORY] = view[2, :, :HISTORY]
        if align:
            target[4, :, :HISTORY] = view[1, :, 1:4]
        views.append(view)
        expected.append(golden)
    address = torch.tensor([view.data_ptr() for view in views], dtype=torch.int64, device=npu)
    page_stride, channel_stride, token_stride = (
        torch.tensor([view.stride(axis) for view in views], dtype=torch.int64, device=npu) for axis in range(3)
    )
    source, lengths, finals, boundary, boundary_len = (
        torch.tensor([*values, *([0] * (batch - 4))], dtype=torch.int32, device=npu)
        for values in ([1, 2, 0, 5], [4, 1, 3, 0], [3, 2, 6, 5], [4, 0, 0, 0], [2, 0, 0, 0])
    )
    compact_conv_state(
        views[0],
        address,
        page_stride,
        channel_stride,
        token_stride,
        source,
        lengths,
        finals,
        boundary,
        boundary_len,
        conv_history_len=HISTORY,
        align_mode=align,
        dim_stride=views[0].stride(1),
        null_block_id=0,
    )
    for backing, golden in zip(backings, expected):
        torch.testing.assert_close(backing, golden, atol=0, rtol=0)
