# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import torch
import torch_npu

import vllm_ascend.ops.register_custom_ops  # noqa: F401
from vllm_ascend.ops.triton.static_quantize import quantize


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows,width", [(0, 17), (1, 2048), (6, 8192), (9, 257)])
@pytest.mark.parametrize("channel", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_projection_quantize_matches_cann(dtype, rows, width, channel, strided):
    # A strided slice with live, nonuniform per-channel parameters. Include
    # ties, saturation and non-finite values to exercise the rounding contract.
    values = torch.tensor(
        [-float("inf"), -128.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127.5, float("inf"), float("nan")],
        device="npu",
        dtype=dtype,
    )
    index = torch.arange(rows * width * 2, device="npu") % len(values)
    x = values[index].reshape(rows, width * 2)[:, ::2]
    reciprocal = (torch.arange(2 * width, device="npu") % 3 + 1).to(dtype)[::2]
    offset = (torch.arange(2 * width, device="npu") % 5 - 2).to(dtype)[::2]
    if not strided:
        x, reciprocal, offset = x.contiguous(), reciprocal.contiguous(), offset.contiguous()
    if not channel:
        reciprocal, offset = reciprocal[:1], offset[:1]
    expected = torch_npu.npu_quantize(
        x, reciprocal.expand(width).contiguous(), offset.expand(width).contiguous(), torch.qint8, -1, False
    )
    with patch.object(torch_npu, "npu_quantize", side_effect=AssertionError("unexpected native fallback")):
        actual = quantize(x, reciprocal, offset)
    assert actual.dtype == torch.int8 and actual.is_contiguous()
    assert torch.equal(actual, expected)


def test_projection_quantize_replay_reads_live_parameters():
    x = torch.ones((6, 2048), dtype=torch.bfloat16, device="npu")
    reciprocal = torch.ones(2048, dtype=x.dtype, device=x.device)
    offset = torch.zeros_like(reciprocal)
    op = torch.ops.vllm.ascend_triton_projection_quantize
    op(x, reciprocal, offset)
    graph = torch.npu.NPUGraph()
    with (
        patch.object(torch_npu, "npu_quantize", side_effect=AssertionError("unexpected native fallback")),
        torch.npu.graph(graph),
    ):
        out = op(x, reciprocal, offset)
    for value in (2, 5):
        reciprocal.fill_(value)
        offset.fill_(1)
        graph.replay()
        expected = torch_npu.npu_quantize(x, reciprocal, offset, torch.qint8, -1, False)
        assert torch.equal(out, expected)
