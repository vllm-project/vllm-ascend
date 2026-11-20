# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused engram_gate equivalence on A2/A3 (guards ENGRAM_UNROTATED_GATE enablement).

After ENGRAM_UNROTATED_GATE was extended from A5-only to A2/A3, 910B
unrotated DeepSeek V4.1 Engram checkpoints take the fused triton kernel
(`fused_engram_gate`) instead of the bare FP32 `engram_gate`. These tests pin
that the two paths agree numerically on 910B. Skipped on non-910B devices
(A5 already runs the fused path natively).
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.engram.common import engram_gate
from vllm_ascend.ops.triton.engram_gate import fused_engram_gate


def _is_910b() -> bool:
    return "910b" in torch.npu.get_device_name(0).lower()


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("hc_mult", [1, 4])
@pytest.mark.parametrize("num_tokens", [2, 3])
@torch.inference_mode()
def test_fused_engram_gate_matches_bare_unrotated(dim, hc_mult, num_tokens):
    if not _is_910b():
        pytest.skip("Fused engram_gate equivalence is an A2/A3 (910B) concern")
    torch.manual_seed(0)
    dev = "npu"
    hidden = torch.randn(num_tokens, hc_mult, dim, dtype=torch.bfloat16, device=dev)
    key = torch.randn(num_tokens, hc_mult, dim, dtype=torch.bfloat16, device=dev)
    value = torch.randn(num_tokens, dim, dtype=torch.bfloat16, device=dev)
    q_weight = torch.randn(hc_mult, dim, dtype=torch.bfloat16, device=dev)
    k_weight = torch.randn(hc_mult, dim, dtype=torch.bfloat16, device=dev)
    mask = torch.ones(num_tokens, dtype=torch.bool, device=dev)
    eps = 1e-6

    # fused wants kv = [T, hc_mult*dim + dim] with key then value, contiguous.
    kv = torch.cat([key.reshape(num_tokens, hc_mult * dim), value], dim=-1).contiguous()
    fused = fused_engram_gate(hidden.contiguous(), kv, q_weight, k_weight, mask, eps)

    # bare takes key/value split + channel_weight = q*k (engram/layer.py:53).
    channel_weight = (q_weight.float() * k_weight.float()).to(torch.bfloat16)
    bare = engram_gate(hidden, key, value, channel_weight, None, mask, eps)

    assert fused.shape == bare.shape
    assert fused.dtype == bare.dtype
    torch.testing.assert_close(fused, bare, rtol=1e-2, atol=1e-2)
