# SPDX-License-Identifier: Apache-2.0
"""fused_qknorm_rope -- Ascend release build (self-contained).

Independent publishable operator: RMSNorm + NeoX RoPE fused into one SIMD
Triton kernel for Ascend (A5 / Ascend 950, Triton-Ascend backend), with
in-kernel V passthrough. No dependency on inference/kernel.py or
_npu_adapt.apply(); import and call ``fused_qknorm_rope`` directly.

Numerical contract: this is an Ascend-specific SIMD implementation (vector
tree reduction for RMSNorm). RoPE rotation angles are read from the
caller-provided cos/sin cache (per-layer rope_theta / rope_scaling are baked
into that cache by the caller). It is NOT bit-identical to the CUDA per-lane
FMA/XOR-tree reference; it matches it within RMSNorm/RoPE tolerance (see
tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_fused_qknorm_rope.py).

Interface (same as the original):
    q, k, v = fused_qknorm_rope(qkv, q_weight, k_weight, cos, sin, positions,
                                head_dim=192, num_q_heads=16, num_kv_heads=1,
                                rotary_pairs=96, eps=1e-5, norm_weight_bias=1.0)
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

HEAD_DIM: int = 192
EPS: float = 1e-5
NORM_WEIGHT_BIAS: float = 1.0
# tl.arange tile extents must be powers of two: the 192-lane head dim pads
# to BLOCK_D and the up-to-96 rotary pairs pad to BLOCK_R.
BLOCK_D = 256
BLOCK_R = 128


# TODO: This will be deleted after qkv_rmsnorm_rope_simt supported head_dim=192
# Use a fixed tokens-per-program tile.
@triton.jit(do_not_specialize=["tokens"])
def _split_qkv_norm_rope_192_kernel(
    qkv_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    tokens,
    stride_qkv_token,
    stride_q_out_token,
    stride_k_out_token,
    stride_v_out_token,
    stride_cos,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    EPS: tl.constexpr,
    NORM_WEIGHT_BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """One program per BLOCK_M tokens, loops over all Q/K heads inside.

    Split into two separate loops (Q heads, then K heads) so that:
      - q_weight / k_weight are each loaded once, outside the loops
      - RoPE shared data (angle, cos, sin, idx) hoisted outside both loops
      - No runtime is_q check, no tl.where, no complementary store masks

    V passthrough (formerly _copy_v_192_kernel) rides along the K loop: V is
    the packed head right after K in qkv, so the same tile traversal copies it
    as raw BF16 -- saving one kernel launch per call.
    """
    pid_m = tl.program_id(0)

    offs_t = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    t_mask = offs_t < tokens

    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    pairs = tl.arange(0, BLOCK_R)
    pair_mask = pairs < ROTARY_PAIRS

    # RoPE angles come from the caller-provided cos/sin cache: the module
    # builds it per layer with the correct rope_theta / rope_scaling, so the
    # kernel must read it instead of recomputing frequencies from a
    # hardcoded theta (previously theta=1e4 was baked in, which corrupted
    # every layer configured with a different rope_theta, e.g. 1e7).
    position = tl.load(positions_ptr + offs_t, mask=t_mask, other=0)
    rope_row = position[:, None] * stride_cos + pairs[None, :]
    rope_mask = t_mask[:, None] & pair_mask[None, :]
    cos_value = tl.load(cos_ptr + rope_row, mask=rope_mask, other=0.0).to(tl.float32)
    sin_value = tl.load(sin_ptr + rope_row, mask=rope_mask, other=0.0).to(tl.float32)

    idx0 = tl.broadcast_to(pairs[None, :], (BLOCK_M, BLOCK_R))
    idx1 = idx0 + ROTARY_PAIRS

    q_weight = tl.load(q_weight_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32) + NORM_WEIGHT_BIAS
    k_weight = tl.load(k_weight_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32) + NORM_WEIGHT_BIAS

    non_rot_mask = (dims >= 2 * ROTARY_PAIRS) & dim_mask
    rot_store_mask = t_mask[:, None] & pair_mask[None, :]
    non_rot_store_mask = non_rot_mask[None, :] & t_mask[:, None]
    load_mask = t_mask[:, None] & dim_mask[None, :]

    for q_head in tl.range(NUM_Q_HEADS):
        head_base = offs_t[:, None] * stride_qkv_token + q_head * HEAD_DIM
        values = tl.load(
            qkv_ptr + head_base + dims[None, :],
            mask=t_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        sum_squares = tl.sum(values * values, axis=1)
        inverse_rms = tl.rsqrt(sum_squares / HEAD_DIM + EPS)

        normalized = (values * inverse_rms[:, None] * q_weight[None, :]).to(tl.bfloat16)
        normalized_f32 = normalized.to(tl.float32)

        value0 = tl.gather(normalized_f32, idx0, axis=1)
        value1 = tl.gather(normalized_f32, idx1, axis=1)
        rotated0 = (value0 * cos_value - value1 * sin_value).to(tl.bfloat16)
        rotated1 = tl.fma(value0, sin_value, value1 * cos_value).to(tl.bfloat16)

        q_base = offs_t[:, None] * stride_q_out_token + q_head * HEAD_DIM
        tl.store(q_out_ptr + q_base + idx0, rotated0, mask=rot_store_mask)
        tl.store(q_out_ptr + q_base + idx1, rotated1, mask=rot_store_mask)
        tl.store(q_out_ptr + q_base + dims[None, :], normalized, mask=non_rot_store_mask)

    for kv_head in tl.range(NUM_KV_HEADS):
        packed_head = NUM_Q_HEADS + kv_head
        head_base = offs_t[:, None] * stride_qkv_token + packed_head * HEAD_DIM
        values = tl.load(
            qkv_ptr + head_base + dims[None, :],
            mask=t_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        sum_squares = tl.sum(values * values, axis=1)
        inverse_rms = tl.rsqrt(sum_squares / HEAD_DIM + EPS)

        normalized = (values * inverse_rms[:, None] * k_weight[None, :]).to(tl.bfloat16)
        normalized_f32 = normalized.to(tl.float32)

        value0 = tl.gather(normalized_f32, idx0, axis=1)
        value1 = tl.gather(normalized_f32, idx1, axis=1)
        rotated0 = (value0 * cos_value - value1 * sin_value).to(tl.bfloat16)
        rotated1 = tl.fma(value0, sin_value, value1 * cos_value).to(tl.bfloat16)

        k_base = offs_t[:, None] * stride_k_out_token + kv_head * HEAD_DIM
        tl.store(k_out_ptr + k_base + idx0, rotated0, mask=rot_store_mask)
        tl.store(k_out_ptr + k_base + idx1, rotated1, mask=rot_store_mask)
        tl.store(k_out_ptr + k_base + dims[None, :], normalized, mask=non_rot_store_mask)

        # V passthrough: V must retain its original bits.
        v_head = packed_head + NUM_KV_HEADS
        v_in_base = offs_t[:, None] * stride_qkv_token + v_head * HEAD_DIM
        v_value = tl.load(qkv_ptr + v_in_base + dims[None, :], mask=load_mask)
        v_out_base = offs_t[:, None] * stride_v_out_token + kv_head * HEAD_DIM
        tl.store(v_out_ptr + v_out_base + dims[None, :], v_value, mask=load_mask)


def _fused_qknorm_rope_impl(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_pairs: int,
    eps: float = EPS,
    norm_weight_bias: float = NORM_WEIGHT_BIAS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_dim != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}, got {head_dim}")

    tokens = qkv.shape[0]
    q_out = torch.empty((tokens, num_q_heads * HEAD_DIM), device=qkv.device, dtype=qkv.dtype)
    k_out = torch.empty((tokens, num_kv_heads * HEAD_DIM), device=qkv.device, dtype=qkv.dtype)
    v_out = torch.empty((tokens, num_kv_heads * HEAD_DIM), device=qkv.device, dtype=qkv.dtype)
    if tokens == 0:
        return q_out, k_out, v_out

    BLOCK_M = 16
    grid = (triton.cdiv(tokens, BLOCK_M),)

    _split_qkv_norm_rope_192_kernel[grid](
        qkv,
        q_out,
        k_out,
        v_out,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        tokens,
        qkv.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        v_out.stride(0),
        cos.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        ROTARY_PAIRS=rotary_pairs,
        EPS=float(eps),
        NORM_WEIGHT_BIAS=float(norm_weight_bias),
        HEAD_DIM=HEAD_DIM,
        BLOCK_D=BLOCK_D,
        BLOCK_R=BLOCK_R,
        BLOCK_M=BLOCK_M,
    )
    return q_out, k_out, v_out


def _fused_qknorm_rope_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_pairs: int,
    eps: float = EPS,
    norm_weight_bias: float = NORM_WEIGHT_BIAS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = qkv.shape[0]
    q = torch.empty((tokens, num_q_heads * head_dim), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty((tokens, num_kv_heads * head_dim), device=qkv.device, dtype=qkv.dtype)
    v = torch.empty((tokens, num_kv_heads * head_dim), device=qkv.device, dtype=qkv.dtype)
    return q, k, v


direct_register_custom_op(
    op_name="ascend_step5_fused_qknorm_rope",
    op_func=_fused_qknorm_rope_impl,
    mutates_args=[],
    fake_impl=_fused_qknorm_rope_fake,
)


def fused_qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_pairs: int,
    eps: float = EPS,
    norm_weight_bias: float = NORM_WEIGHT_BIAS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.ascend_step5_fused_qknorm_rope(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_pairs=rotary_pairs,
        eps=eps,
        norm_weight_bias=norm_weight_bias,
    )
