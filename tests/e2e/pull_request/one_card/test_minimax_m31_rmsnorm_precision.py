# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3RMSNorm
from vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache import minimax_qknorm_rope_cache


def reference(x, weight, eps):
    return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), 1 + weight.float(), eps).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_m31_native_rmsnorm_precision(dtype, with_residual):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(5, 128, generator=generator).to(dtype)
    residual = torch.randn(5, 128, generator=generator).to(dtype)
    norm = MiniMaxM3RMSNorm(128).to(device="npu", dtype=dtype)
    weight = torch.linspace(-0.1, 0.1, 128).to(dtype)
    norm.weight.copy_(weight.npu())
    expected_input = x + residual if with_residual else x
    expected = reference(expected_input, weight, norm.variance_epsilon)
    x_npu, residual_npu = x.npu(), residual.npu()
    if with_residual:
        output, combined = norm(x_npu, residual_npu)
        torch.testing.assert_close(combined.cpu(), expected_input, atol=0, rtol=0)
    else:
        output = norm(x_npu)
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 1e-2, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(output.cpu(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(x_npu.cpu(), x, atol=0, rtol=0)
    torch.testing.assert_close(residual_npu.cpu(), residual, atol=0, rtol=0)


@pytest.mark.parametrize("separate_index", [False, True])
@torch.inference_mode()
def test_m31_fused_and_unfused_norm_agree(separate_index):
    # Identity RoPE isolates the normalization contract. Small offsets expose
    # prematurely rounded gamma values while retaining real NPU kernel execution.
    generator = torch.Generator().manual_seed(17)
    tokens, head_dim, eps = 4, 128, 1e-6
    packed = torch.randn(tokens, 5 * head_dim, generator=generator).bfloat16()
    weights = [torch.linspace(-0.1, 0.1, head_dim).roll(i * 7).bfloat16() for i in range(4)]
    cos_sin = torch.cat((torch.ones(1, 64), torch.zeros(1, 64)), dim=-1).bfloat16().npu()
    positions = torch.zeros(tokens, dtype=torch.int64, device="npu")
    slots = torch.arange(tokens, dtype=torch.int64, device="npu")
    key_cache = torch.zeros(1, 16, 1, head_dim, dtype=torch.bfloat16, device="npu")
    value_cache = torch.zeros_like(key_cache)
    index_cache = torch.zeros(1, 16, head_dim, dtype=torch.bfloat16, device="npu")
    packed_npu = packed.npu()
    q, iq = minimax_qknorm_rope_cache(
        packed_npu[:, : 3 * head_dim] if separate_index else packed_npu,
        cos_sin,
        positions,
        *[w.npu() for w in weights],
        key_cache,
        value_cache,
        index_cache,
        slots,
        slots,
        tokens,
        1,
        1,
        1,
        eps,
        index_packed=packed_npu[:, 3 * head_dim :] if separate_index else None,
    )
    actual = [q, key_cache[0, :tokens, 0], iq, index_cache[0, :tokens]]
    parts = packed.split(head_dim, dim=-1)
    for part, weight, fused in zip((0, 1, 3, 4), weights, actual):
        norm = MiniMaxM3RMSNorm(head_dim, eps).to(device="npu", dtype=torch.bfloat16)
        norm.weight.copy_(weight.npu())
        unfused = norm(parts[part].npu())
        gold = reference(parts[part], weight, eps)
        torch.testing.assert_close(fused.cpu(), gold, atol=0.016, rtol=0.01)
        torch.testing.assert_close(unfused.cpu(), gold, atol=0.016, rtol=0.01)
        torch.testing.assert_close(fused, unfused, atol=0.016, rtol=0.01)
    torch.testing.assert_close(value_cache[0, :tokens, 0].cpu(), parts[2], atol=0, rtol=0)
