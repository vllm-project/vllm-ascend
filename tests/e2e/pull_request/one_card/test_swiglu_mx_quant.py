# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
import torch_npu

pytestmark = pytest.mark.skip(
    reason="CI torch_npu 2.10.0.post4 does not support swiglu_mode=2; re-enable after the CI runtime is upgraded"
)


def fused(x, limit, alpha, bias):
    return torch_npu.npu_swiglu_mx_quant(
        x,
        activate_left=True,
        swiglu_mode=2,
        clamp_limit=limit,
        glu_alpha=alpha,
        glu_bias=bias,
        dst_type=torch.float8_e4m3fn,
        scale_alg=1,
    )


def dequantize(q, scale):
    q = q.cpu().float()
    if q.numel() == 0:
        return q
    exponent = scale.cpu().reshape(q.shape[0], -1).float() - 127
    return q * torch.exp2(exponent).repeat_interleave(32, dim=-1)[:, : q.shape[-1]]


@pytest.mark.parametrize("rows", [0, 1, 16, 128])
@pytest.mark.parametrize("width", [128, 2048])
@pytest.mark.parametrize("limit,alpha,bias", [(7.0, 1.702, 1.0), (5.0, 1.5, 0.5)])
@torch.inference_mode()
def test_clipped_swiglu_mx_quant_numerics(rows, width, limit, alpha, bias):
    if not any(tag in torch.npu.get_device_name(0).lower() for tag in ("950", "910_95")):
        pytest.skip("The CEIL-scale fused activation is an A5 path")
    torch.manual_seed(19)
    x = torch.randn(rows, 2 * width, device="npu", dtype=torch.bfloat16) * 4
    actual, scale = fused(x, limit, alpha, bias)
    gate, up = x.cpu().float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    reference = gate * torch.sigmoid(alpha * gate) * (up + bias)
    assert actual.shape == reference.shape
    assert actual.dtype == torch.float8_e4m3fn
    # E4M3 rounding has at most half an ULP (6.25% for normal values).
    torch.testing.assert_close(dequantize(actual, scale), reference, rtol=0.07, atol=0.02)


@torch.inference_mode()
def test_clipped_swiglu_mx_quant_graph_replay():
    if not any(tag in torch.npu.get_device_name(0).lower() for tag in ("950", "910_95")):
        pytest.skip("The CEIL-scale fused activation is an A5 path")
    x = torch.randn(128, 4096, device="npu", dtype=torch.bfloat16)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            fused(x, 7.0, 1.702, 1.0)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual, scale = fused(x, 7.0, 1.702, 1.0)
    pointers = (actual.data_ptr(), scale.data_ptr())
    for _ in range(3):
        x.copy_(torch.randn_like(x) * 4)
        expected, expected_scale = fused(x, 7.0, 1.702, 1.0)
        graph.replay()
        torch.npu.synchronize()
        assert (actual.data_ptr(), scale.data_ptr()) == pointers
        torch.testing.assert_close(actual.cpu().float(), expected.cpu().float(), rtol=0, atol=0)
        torch.testing.assert_close(scale.cpu(), expected_scale.cpu(), rtol=0, atol=0)
