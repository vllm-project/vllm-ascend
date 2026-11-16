# SPDX-License-Identifier: Apache-2.0
"""Single-operator accuracy test for the Step5 fused_qknorm_rope Triton op.

Reference: pure-torch mirror of the kernel numerics (fp32 RMSNorm reduction,
bf16 rounding before rotation, fp32 rotation, bf16 store). Q/K use the BF16
tolerance (rtol = atol = 2^-6, >= 99% of elements within the mixed
tolerance, max abs error <= 1); V is a raw copy and must be bit-exact.

Real Step5 shape first (DSA layer, tp=1: 16 Q heads, 1 KV head, head_dim
192, rotary_pairs 96, rope_theta 1e7), then GQA / partial-RoPE / bias
variants. Token counts cover 1 (single decode), odd tiles and 4096.
"""

import gc

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.step5.fused_qknorm_rope import fused_qknorm_rope

HEAD_DIM = 192
RTOL = 2.0**-6
ATOL = 2.0**-6
MAX_ABS_ERR = 1.0


def build_rope_cache(max_pos, rotary_pairs, theta, device):
    """NeoX cos/sin cache halves, cos_sin_cache.chunk(2, dim=-1) layout."""
    inv_freq = theta ** (-torch.arange(0, rotary_pairs, dtype=torch.float32, device=device) / rotary_pairs)
    pos = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    return cache[:, :rotary_pairs].contiguous(), cache[:, rotary_pairs:].contiguous()


def torch_ref(
    qkv, q_weight, k_weight, cos, sin, positions, *, num_q_heads, num_kv_heads, rotary_pairs, eps, norm_weight_bias
):
    tokens = qkv.shape[0]
    qkv_f = qkv.float()

    def norm_heads(x_f, weight):
        inv_rms = torch.rsqrt((x_f * x_f).sum(-1) / HEAD_DIM + eps)
        return (x_f * inv_rms.unsqueeze(-1) * (weight.float() + norm_weight_bias)).to(torch.bfloat16)

    q = norm_heads(qkv_f[:, : num_q_heads * HEAD_DIM].view(tokens, num_q_heads, HEAD_DIM), q_weight)
    k = norm_heads(
        qkv_f[:, num_q_heads * HEAD_DIM : (num_q_heads + num_kv_heads) * HEAD_DIM].view(tokens, num_kv_heads, HEAD_DIM),
        k_weight,
    )
    v = qkv[:, (num_q_heads + num_kv_heads) * HEAD_DIM :].clone()

    cos_f = cos[positions].float().unsqueeze(1)  # [tokens, 1, rotary_pairs]
    sin_f = sin[positions].float().unsqueeze(1)

    def rope(x_bf16):
        x = x_bf16.float()
        x1 = x[..., :rotary_pairs]
        x2 = x[..., rotary_pairs : 2 * rotary_pairs]
        rotated = torch.cat(
            [(x1 * cos_f - x2 * sin_f).to(torch.bfloat16), (x1 * sin_f + x2 * cos_f).to(torch.bfloat16)],
            dim=-1,
        )
        return torch.cat([rotated, x_bf16[..., 2 * rotary_pairs :]], dim=-1)

    return rope(q).reshape(tokens, -1), rope(k).reshape(tokens, -1), v.reshape(tokens, -1)


def assert_bf16_close(actual, ref, name):
    a = actual.float()
    r = ref.float()
    diff = (a - r).abs()
    within = diff <= ATOL + RTOL * r.abs()
    frac = within.float().mean().item()
    assert frac >= 0.99, f"{name}: only {frac:.4%} elements within BF16 tolerance (max abs err {diff.max().item():.6f})"
    assert diff.max().item() <= MAX_ABS_ERR, f"{name}: max abs err {diff.max().item():.6f} exceeds {MAX_ABS_ERR}"


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
@pytest.mark.parametrize(
    "num_q_heads, num_kv_heads, rotary_pairs, tokens, theta, bias",
    [
        # Real Step5 DSA-layer shape (sparse_indexer_num_heads=16,
        # sparse_indexer_num_k_heads=1, head_dim=192, full 192-lane RoPE,
        # per-layer rope_theta 1e7 - the case a hardcoded theta broke).
        (16, 1, 96, 1, 1e7, 1.0),
        (16, 1, 96, 4096, 1e7, 1.0),
        # Plain (non zero-centered) norm weights and another theta.
        (16, 1, 96, 333, 1e4, 0.0),
        # Dense-attention GQA shape.
        (32, 8, 96, 257, 1e7, 1.0),
        # Partial RoPE (rotary_dim 128 < head_dim 192): un-rotated tail.
        (16, 1, 64, 129, 1e6, 1.0),
    ],
)
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_fused_qknorm_rope_accuracy(num_q_heads, num_kv_heads, rotary_pairs, tokens, theta, bias, eps):
    torch.manual_seed(0)
    device = "npu"
    max_pos = 8192
    q_width = num_q_heads * HEAD_DIM
    kv_width = num_kv_heads * HEAD_DIM
    qkv = (torch.randn(tokens, q_width + 2 * kv_width, device=device) * 2).to(torch.bfloat16)
    q_weight = (torch.randn(HEAD_DIM, device=device) * 0.2 + 1.0).to(torch.bfloat16)
    k_weight = (torch.randn(HEAD_DIM, device=device) * 0.2 + 1.0).to(torch.bfloat16)
    cos, sin = build_rope_cache(max_pos, rotary_pairs, theta, device)
    # Positions near the cache top exercise large row offsets / int64 loads.
    positions = torch.randint(max_pos - tokens, max_pos, (tokens,), device=device, dtype=torch.int64)

    q, k, v = fused_qknorm_rope(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_pairs=rotary_pairs,
        eps=eps,
        norm_weight_bias=bias,
    )

    q_ref, k_ref, v_ref = torch_ref(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_pairs=rotary_pairs,
        eps=eps,
        norm_weight_bias=bias,
    )
    assert_bf16_close(q, q_ref, "q")
    assert_bf16_close(k, k_ref, "k")
    # V passthrough must retain the original bits.
    torch.testing.assert_close(v, v_ref, rtol=0, atol=0, msg="v passthrough is not bit-exact")

    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
def test_fused_qknorm_rope_zero_tokens():
    device = "npu"
    cos, sin = build_rope_cache(64, 96, 1e7, device)
    qkv = torch.zeros(0, (16 + 2) * HEAD_DIM, dtype=torch.bfloat16, device=device)
    q, k, v = fused_qknorm_rope(
        qkv,
        torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device),
        torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device),
        cos,
        sin,
        torch.zeros(0, dtype=torch.int64, device=device),
        head_dim=HEAD_DIM,
        num_q_heads=16,
        num_kv_heads=1,
        rotary_pairs=96,
    )
    assert q.shape == (0, 16 * HEAD_DIM)
    assert k.shape == (0, HEAD_DIM)
    assert v.shape == (0, HEAD_DIM)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
def test_fused_qknorm_rope_rejects_other_head_dim():
    device = "npu"
    cos, sin = build_rope_cache(64, 96, 1e7, device)
    with pytest.raises(ValueError, match="head_dim"):
        fused_qknorm_rope(
            torch.zeros(4, 18 * 128, dtype=torch.bfloat16, device=device),
            torch.ones(128, dtype=torch.bfloat16, device=device),
            torch.ones(128, dtype=torch.bfloat16, device=device),
            cos,
            sin,
            torch.arange(4, device=device),
            head_dim=128,
            num_q_heads=16,
            num_kv_heads=1,
            rotary_pairs=64,
        )
