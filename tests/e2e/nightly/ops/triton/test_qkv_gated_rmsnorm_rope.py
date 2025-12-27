import gc
from typing import Tuple

import pytest
import torch

from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope import \
    split_qkv_gated_rmsnorm_rope_impl
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DTYPES = [torch.bfloat16]
HEAD_SIZES = [128, 256]
ROTARY_DIMS = [32, 64]
NUM_Q_HEADS = [4]
NUM_K_HEADS = [1]
NUM_TOKENS = [32, 64, 256, 1024]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
EPS = [1e-6]
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2


def _qkv_rmsnorm_rope_with_gate_naive(
    qkv: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
    q_weight: torch.Tensor, k_weight: torch.Tensor, q_size: int, kv_size: int,
    num_heads: int, num_kv_heads: int, head_size: int, rotary_dim: int,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
    orig_dtype = qkv.dtype
    orig_shape = q_gate.shape[:-1]

    q_gate_reshaped = q_gate.view(*orig_shape, num_heads, -1)
    q, gate = torch.chunk(q_gate_reshaped, 2, dim=-1)

    gate = gate.reshape(*orig_shape, q_size)
    k = k.view(*orig_shape, num_kv_heads, head_size)

    def _rms_norm(x: torch.Tensor, weight: torch.Tensor,
                  eps: float) -> torch.Tensor:
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + eps)
        return (x / rms) * (weight.to(torch.float32) + 1.0)

    q_norm = _rms_norm(q, q_weight, eps)
    k_norm = _rms_norm(k, k_weight, eps)

    def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                    rotary_dim: int) -> torch.Tensor:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]

        x1 = x_rot[..., :rotary_dim // 2]
        x2 = x_rot[..., rotary_dim // 2:]
        x_rot_rotated = torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(-2).to(torch.float32)
        sin = sin.unsqueeze(-2).to(torch.float32)

        x_rot = x_rot * cos + x_rot_rotated * sin

        return torch.cat([x_rot, x_pass], dim=-1)

    q_rope = _apply_rope(q_norm, sin, cos,
                         rotary_dim).reshape(*orig_shape,
                                             q_size).to(orig_dtype)
    k_rope = _apply_rope(k_norm, sin, cos,
                         rotary_dim).reshape(*orig_shape,
                                             kv_size).to(orig_dtype)

    return q_rope, gate, k_rope, v


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads", NUM_Q_HEADS)
@pytest.mark.parametrize("num_k_heads", NUM_K_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_qkv_rmsnorm_rope_fusion_with_gate_triton_kernel(
    num_tokens: int,
    num_q_heads: int,
    num_k_heads: int,
    head_size: int,
    rotary_dim: int,
    eps: float,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()
    q_size = head_size * num_q_heads
    kv_size = head_size * num_k_heads
    sin = torch.randn(num_tokens, rotary_dim // 2, dtype=dtype, device=device)
    cos = torch.randn(num_tokens, rotary_dim // 2, dtype=dtype, device=device)
    sin = sin.repeat(1, 2)
    cos = cos.repeat(1, 2)

    qkv = torch.randn(num_tokens,
                      q_size * 2 + kv_size * 2,
                      dtype=dtype,
                      device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)

    q_trt, gate_trt, k_trt, v_trt = split_qkv_gated_rmsnorm_rope_impl(
        qkv, sin, cos, q_weight, k_weight, q_size, kv_size, num_q_heads,
        num_k_heads, head_size, rotary_dim, eps)
    q_gold, gate_gold, k_gold, v_gold = _qkv_rmsnorm_rope_with_gate_naive(
        qkv, sin, cos, q_weight, k_weight, q_size, kv_size, num_q_heads,
        num_k_heads, head_size, rotary_dim, eps)
    # Compare the results.
    torch.testing.assert_close(q_trt,
                               q_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    torch.testing.assert_close(gate_trt,
                               gate_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    torch.testing.assert_close(k_trt,
                               k_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    torch.testing.assert_close(v_trt,
                               v_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
