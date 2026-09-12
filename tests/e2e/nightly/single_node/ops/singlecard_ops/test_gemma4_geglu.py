# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma4 expert activation, including TP partitions and ACLGraph replay."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import is_950
from vllm_ascend.device.device_op import A5DeviceAdaptor

pytestmark = pytest.mark.skipif(not is_950(), reason="A5 GeGlu implementation")


def _reference(x):
    gate, up = x.float().chunk(2, dim=-1)
    return (torch.nn.functional.gelu(gate, approximate="tanh") * up).to(x.dtype)


def _assert_close(actual, expected):
    tolerance = max(2 * torch.finfo(actual.dtype).eps, 1e-5)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("width", [176, 352, 704])
@pytest.mark.parametrize("rows", [0, 1, 8, 32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_gemma4_geglu(width, rows, dtype):
    generator = torch.Generator().manual_seed(0)
    x = (torch.randn(rows, 2 * width, generator=generator) * 3).to(dtype).npu()
    actual = A5DeviceAdaptor.gelu_tanh_and_mul(x)
    assert actual.shape == (rows, width)
    _assert_close(actual, _reference(x))


@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_gemma4_geglu_batched_input(strided):
    x = torch.randn(2, 3, 2816, dtype=torch.bfloat16).npu()
    x = x[..., ::2] if strided else x[..., :1408].contiguous()
    actual = A5DeviceAdaptor.gelu_tanh_and_mul(x)
    _assert_close(actual, _reference(x))


@torch.inference_mode()
def test_gemma4_geglu_mxfp4_graph_replay():
    x = torch.randn(8, 1408, dtype=torch.bfloat16).npu()

    def compute():
        activated = A5DeviceAdaptor.gelu_tanh_and_mul(x)
        return torch_npu.npu_dynamic_mx_quant(activated, dst_type=torch_npu.float4_e2m1fn_x2)

    for _ in range(3):
        compute()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_quant, graph_scale = compute()
    # A second input checks that replay reads current data, not capture values.
    x.copy_(torch.randn_like(x))
    graph.replay()
    eager_quant, eager_scale = compute()
    torch.npu.synchronize()
    torch.testing.assert_close(graph_quant.view(torch.uint8), eager_quant.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(graph_scale.view(torch.uint8), eager_scale.view(torch.uint8), rtol=0, atol=0)
