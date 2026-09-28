# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query


def reference(query):
    scale = (query.float().abs().amax(-1) / 127.0).half().clamp_min_(2.0**-24)
    quantized = (query.float() / scale.float().unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def assert_quantized_equal(query, expected):
    actual = quantize_indexer_query(query)
    for output, ref in zip(actual, expected):
        assert output.is_contiguous()
        torch.testing.assert_close(output.cpu(), ref.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("tokens", [0, 1, 3, 32, 129, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_quantize_indexer_query(heads, tokens, dtype):
    torch.manual_seed(41)
    query = torch.randn(tokens, heads, 128, dtype=dtype, device="npu")
    original = query.clone()
    assert_quantized_equal(query, reference(query))
    torch.testing.assert_close(query, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_quantize_indexer_query_rounding_and_scale(dtype):
    # Scale=1 exposes positive and negative ties. Other rows cover zero,
    # FP16 subnormal scales, scale rounding and FP16 scale overflow.
    query = torch.zeros(1, 32, 128, dtype=torch.float32)
    query[0, 0, :10] = torch.tensor([127, -127, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5])
    query[0, 2] = query[0, 0] * 2.0**-24
    query[0, 3] = query[0, 0] * 2.0**-25
    query[0, 4] = torch.linspace(-1.001, 1.001, 128)
    query[0, 5] = torch.finfo(dtype).max
    query = query.to(dtype).npu()
    expected = reference(query)
    torch.testing.assert_close(
        expected[0][0, 0, :10].cpu(), torch.tensor([127, -127, 0, 0, 2, -2, 2, -2, 4, -4], dtype=torch.int8)
    )
    assert_quantized_equal(query, expected)


@pytest.mark.parametrize("heads", [32, 64])
@torch.inference_mode()
def test_quantize_indexer_query_noncontiguous(heads):
    query = torch.randn(3, heads, 256, dtype=torch.bfloat16, device="npu")[..., ::2]
    assert not query.is_contiguous()
    assert_quantized_equal(query, reference(query))


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("tokens", [3, 129])
@torch.inference_mode()
def test_quantize_indexer_query_graph_replay(heads, tokens):
    query = torch.randn(tokens, heads, 128, dtype=torch.bfloat16, device="npu")
    quantize_indexer_query(query)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = quantize_indexer_query(query)
    pointers = tuple(output.data_ptr() for output in actual)
    for magnitude in (2.0, 1e-6, 0.0):
        query.copy_(torch.randn_like(query) * magnitude)
        expected = reference(query)
        graph.replay()
        torch.npu.synchronize()
        assert tuple(output.data_ptr() for output in actual) == pointers
        for output, ref in zip(actual, expected):
            torch.testing.assert_close(output.cpu(), ref.cpu(), rtol=0, atol=0)
