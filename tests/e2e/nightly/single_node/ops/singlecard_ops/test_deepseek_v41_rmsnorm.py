# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU coverage for the shared compressor and indexer K RMSNorm."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41RMSNorm


def _reference(x, weight, eps):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight


@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("tokens", [0, 1, 32, 4096])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
@pytest.mark.parametrize("scale", [0.0, 1e-4, 1.0])
@torch.inference_mode()
def test_rmsnorm_matches_reference(width, tokens, eps, scale):
    torch.manual_seed(41)
    x = (torch.randn(tokens, width) * scale).bfloat16()
    norm = DeepseekV41RMSNorm(width, eps).npu()
    norm.weight.copy_(torch.randn_like(norm.weight))
    expected = _reference(x, norm.weight.cpu(), eps)
    x_npu = x.npu()
    actual = norm(x_npu)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    torch.testing.assert_close(x_npu.cpu(), x, rtol=0, atol=0)
    # The old equation rounds to BF16 before multiplying by the weight.
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.016, atol=1e-5)


@pytest.mark.parametrize("width", [128, 512])
@torch.inference_mode()
def test_rmsnorm_graph_replay_uses_new_input(width):
    torch.manual_seed(42)
    norm = DeepseekV41RMSNorm(width, 1e-6).npu()
    norm.weight.copy_(torch.randn_like(norm.weight))
    weight = norm.weight.cpu()
    x = torch.randn(32, width, dtype=torch.bfloat16, device="npu")
    norm(x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = norm(x)
    pointers = (x.data_ptr(), actual.data_ptr())
    for scale in (1.0, 1e-4, 0.0):
        updated = (torch.randn(32, width) * scale).bfloat16()
        x.copy_(updated)
        graph.replay()
        torch.npu.synchronize()
        assert (x.data_ptr(), actual.data_ptr()) == pointers
        expected = _reference(updated, weight, norm.eps)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.016, atol=1e-5)
