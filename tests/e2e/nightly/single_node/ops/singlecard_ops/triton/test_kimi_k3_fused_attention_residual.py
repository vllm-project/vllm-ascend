# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.kimi_k3 import fused_attention_residual


@pytest.mark.parametrize("num_tokens", [1, 32])
def test_kimi_k3_fused_attention_residual(num_tokens: int):
    torch.manual_seed(0)
    hidden_size = 7168
    num_residuals = 4
    eps = 1e-5
    prefix_sum = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16).npu()
    block_residual = torch.randn(
        num_tokens,
        num_residuals,
        hidden_size,
        dtype=torch.bfloat16,
    ).npu()
    projection_weight = (torch.randn(hidden_size, dtype=torch.bfloat16) * 0.01).npu()
    norm_weight = torch.randn(hidden_size, dtype=torch.bfloat16).npu()

    actual = fused_attention_residual(
        prefix_sum,
        block_residual,
        projection_weight,
        norm_weight,
        eps,
    )

    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1).float()
    inverse_rms = torch.rsqrt(values.square().mean(-1) + eps)
    scores = torch.matmul(values, (norm_weight * projection_weight).float()) * inverse_rms
    probabilities = scores.softmax(-1)
    expected = torch.sum(probabilities.unsqueeze(-1) * values, dim=1).to(prefix_sum.dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
