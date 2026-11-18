# SPDX-License-Identifier: Apache-2.0
"""A5 shared SiTU numerical and latency regression; requires custom operators."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SituAndMul

from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedSparseSituAndMul


@pytest.mark.parametrize("tokens", [1, 8, 128])
def test_shared_sparse_situ_accuracy(tokens, record_property):
    if not torch.npu.is_available() or not hasattr(torch.ops.custom, "npu_situ_and_mul_sparse"):
        pytest.skip("requires Ascend NPU and sparse SiTU operator")
    torch.manual_seed(7)
    with set_current_vllm_config(VllmConfig()):
        layer = AscendSharedSparseSituAndMul(beta=4.0, linear_beta=25.0).npu()
    x = torch.randn(tokens, 1536, device="npu", dtype=torch.bfloat16)
    expected = SituAndMul.forward_native(layer, x)
    for _ in range(3):
        actual = layer.forward_native(x)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    repeats = 20
    for _ in range(repeats):
        layer.forward_native(x)
    end.record()
    end.synchronize()
    record_property("mean_ms", start.elapsed_time(end) / repeats)
