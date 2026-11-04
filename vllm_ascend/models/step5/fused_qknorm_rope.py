# SPDX-License-Identifier: Apache-2.0
"""fused_qknorm_rope -- Ascend release build (self-contained).

Independent publishable operator: RMSNorm + NeoX RoPE fused into one SIMD
Triton kernel for Ascend (910B, Triton-Ascend backend), with in-kernel V
passthrough. No dependency on inference/kernel.py or _npu_adapt.apply();
import and call ``fused_qknorm_rope`` directly.

Numerical contract: this is an Ascend-specific SIMD implementation (vector
tree reduction for RMSNorm). RoPE rotation angles are read from the
caller-provided cos/sin cache (per-layer rope_theta / rope_scaling are baked
into that cache by the caller). It is NOT bit-identical to the CUDA per-lane
FMA/XOR-tree reference; it matches it within RMSNorm/RoPE tolerance (see
check_precision.py).

Interface (same as the original):
    q, k, v = fused_qknorm_rope(qkv, q_weight, k_weight, cos, sin, positions,
                                head_dim=192, num_q_heads=16, num_kv_heads=1,
                                rotary_pairs=96, eps=1e-5, norm_weight_bias=1.0)
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


HEAD_DIM: int = 192
EPS: float = 1e-5
NORM_WEIGHT_BIAS: float = 1.0


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm})
        for bm in [4, 8, 16, 32]
    ],
    key=["tokens"],
)
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

    grid = lambda meta: (triton.cdiv(tokens, meta["BLOCK_M"]),)
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
        BLOCK_D=256,
        BLOCK_R=128,
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
    q = torch.empty(
        (tokens, num_q_heads * head_dim), device=qkv.device, dtype=qkv.dtype
    )
    k = torch.empty(
        (tokens, num_kv_heads * head_dim), device=qkv.device, dtype=qkv.dtype
    )
    v = torch.empty(
        (tokens, num_kv_heads * head_dim), device=qkv.device, dtype=qkv.dtype
    )
    return q, k, v


from vllm.utils.torch_utils import direct_register_custom_op

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


# ============================================================================
# Fused indexer norm+RoPE for Q/K/Z (DSA sparse indexer)
# ============================================================================

import triton.language.extra.cann.extension as extension


def _npu_vec_core_num() -> int:
    import triton.runtime.driver as driver
    properties = driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )
    return properties["num_vectorcore"]


def _pick_head_chunk(num_heads: int, head_dim: int) -> int:
    budget_cols = 4096
    width = num_heads * head_dim
    if width <= budget_cols:
        return num_heads
    h = max(1, min(num_heads, budget_cols // head_dim))
    return h


@triton.jit
def _norm_rope_tile_npu_chunk(
    x_ptr,
    out_ptr,
    cos_t,
    sin_t,
    token_start,
    stride_x_token,
    stride_out_token,
    seg_base,
    head_off,
    w_hd,
    b_hd,
    w_real,
    w_imag,
    b_real,
    b_imag,
    eps,
    weight_bias,
    tokens,
    CH: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
) -> None:
    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    gr = tl.arange(0, ROWS_PER_PROGRAM * CH)
    tok = gr // CH
    head = gr % CH
    tok_abs = token_start + tok
    row_ok = tok_abs < tokens
    out_base = tok_abs * stride_out_token + head_off + head * head_dim

    x_bp = tl.make_block_ptr(
        base=x_ptr + seg_base,
        shape=(tokens, TOTAL_HEADS * head_dim),
        strides=(stride_x_token, 1),
        offsets=(token_start, head_off),
        block_shape=(ROWS_PER_PROGRAM, CH * head_dim),
        order=(1, 0),
    )
    values = tl.reshape(
        tl.load(x_bp, boundary_check=(0,), padding_option="zero").to(tl.float32),
        (ROWS_PER_PROGRAM * CH, head_dim),
    )

    cos = tl.reshape(
        tl.broadcast_to(cos_t[:, None, :], (ROWS_PER_PROGRAM, CH, BLOCK_R)),
        (ROWS_PER_PROGRAM * CH, BLOCK_R),
    )
    sin = tl.reshape(
        tl.broadcast_to(sin_t[:, None, :], (ROWS_PER_PROGRAM, CH, BLOCK_R)),
        (ROWS_PER_PROGRAM * CH, BLOCK_R),
    )

    sum_sq = tl.sum(values * values, axis=1)
    if SUBTRACT_MEAN:
        mean = tl.sum(values, axis=1) / head_dim
        variance = sum_sq / head_dim - mean * mean
    else:
        mean = tl.zeros((ROWS_PER_PROGRAM * CH,), dtype=tl.float32)
        variance = sum_sq / head_dim
    rstd = tl.rsqrt(variance + eps)
    result_tail = (values - mean[:, None]) * rstd[:, None] * (w_hd + weight_bias)
    if HAS_BIAS:
        result_tail = result_tail + b_hd

    if CH == 1:
        xrow = tok_abs * stride_x_token + head_off + head * head_dim + seg_base
        real_raw = tl.load(
            x_ptr + xrow[:, None] + half[None, :],
            mask=row_ok[:, None] & half_ok[None, :], other=0.0).to(tl.float32)
        imag_raw = tl.load(
            x_ptr + xrow[:, None] + rotary_dim + half[None, :],
            mask=row_ok[:, None] & half_ok[None, :], other=0.0).to(tl.float32)
    else:
        real_raw = extension.extract_slice(values, [0, 0], [ROWS_PER_PROGRAM * CH, BLOCK_R], [1, 1])
        imag_raw = extension.extract_slice(values, [0, rotary_dim], [ROWS_PER_PROGRAM * CH, BLOCK_R], [1, 1])
    real = (real_raw - mean[:, None]) * rstd[:, None] * (w_real + weight_bias)
    imaginary = (imag_raw - mean[:, None]) * rstd[:, None] * (w_imag + weight_bias)
    if HAS_BIAS:
        real = real + b_real
        imaginary = imaginary + b_imag
    real = real.to(x_ptr.dtype.element_ty).to(tl.float32)
    imaginary = imaginary.to(x_ptr.dtype.element_ty).to(tl.float32)
    rotated_first = real * cos - imaginary * sin
    rotated_second = real * sin + imaginary * cos

    out_bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(tokens, TOTAL_HEADS * head_dim),
        strides=(stride_out_token, 1),
        offsets=(token_start, head_off),
        block_shape=(ROWS_PER_PROGRAM, CH * head_dim),
        order=(1, 0),
    )
    # Store through the input precision first: when the output is narrower
    # (index_q lands in e4m3 while the norm+rope math stages in bf16), the
    # double rounding reproduces the unfused pipeline's bits exactly
    # (kernel-store bf16 -> later .to(float8_e4m3fn) cast). Same-dtype
    # outputs take the second cast as a no-op.
    tl.store(
        out_bp,
        tl.reshape(
            result_tail.to(x_ptr.dtype.element_ty).to(out_ptr.dtype.element_ty),
            (ROWS_PER_PROGRAM, CH * head_dim),
        ),
        boundary_check=(0,),
    )
    mask_half = row_ok[:, None] & half_ok[None, :]
    tl.store(
        out_ptr + out_base[:, None] + half[None, :],
        rotated_first.to(x_ptr.dtype.element_ty).to(out_ptr.dtype.element_ty),
        mask=mask_half,
    )
    tl.store(
        out_ptr + out_base[:, None] + rotary_dim + half[None, :],
        rotated_second.to(x_ptr.dtype.element_ty).to(out_ptr.dtype.element_ty),
        mask=mask_half,
    )


@triton.jit
def _norm_rope_tile_npu(
    x_ptr,
    out_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    token_idx,
    token_start,
    token_ok,
    stride_x_token,
    stride_out_token,
    stride_cos,
    seg_base,
    w_all,
    w_real,
    w_imag,
    b_all,
    b_real,
    b_imag,
    eps,
    weight_bias,
    tokens,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    H_CHUNK: tl.constexpr,
    N_FULL: tl.constexpr,
    REM_HEADS: tl.constexpr,
) -> None:
    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    w_hd = extension.extract_slice(tl.reshape(w_all, (1, BLOCK_D)), [0, 0], [1, head_dim], [1, 1])
    if HAS_BIAS:
        b_hd = extension.extract_slice(tl.reshape(b_all, (1, BLOCK_D)), [0, 0], [1, head_dim], [1, 1])
    else:
        b_hd = tl.zeros((1, head_dim), dtype=tl.float32)

    position = tl.load(positions_ptr + token_idx, mask=token_ok, other=0).to(tl.int32)
    cos_t = tl.load(
        cos_ptr + position[:, None] * stride_cos + half[None, :],
        mask=token_ok[:, None] & half_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    extension.compile_hint(cos_t, "mayDiscretememaccess")
    sin_t = tl.load(
        sin_ptr + position[:, None] * stride_cos + half[None, :],
        mask=token_ok[:, None] & half_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    extension.compile_hint(sin_t, "mayDiscretememaccess")

    for c in tl.static_range(N_FULL):
        _norm_rope_tile_npu_chunk(
            x_ptr, out_ptr, cos_t, sin_t, token_start,
            stride_x_token, stride_out_token, seg_base, c * H_CHUNK * head_dim,
            w_hd, b_hd, w_real, w_imag, b_real, b_imag, eps, weight_bias, tokens,
            CH=H_CHUNK, TOTAL_HEADS=num_heads, head_dim=head_dim, rotary_dim=rotary_dim,
            SUBTRACT_MEAN=SUBTRACT_MEAN, HAS_BIAS=HAS_BIAS,
            BLOCK_R=BLOCK_R, ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        )
    if REM_HEADS > 0:
        _norm_rope_tile_npu_chunk(
            x_ptr, out_ptr, cos_t, sin_t, token_start,
            stride_x_token, stride_out_token, seg_base, N_FULL * H_CHUNK * head_dim,
            w_hd, b_hd, w_real, w_imag, b_real, b_imag, eps, weight_bias, tokens,
            CH=REM_HEADS, TOTAL_HEADS=num_heads, head_dim=head_dim, rotary_dim=rotary_dim,
            SUBTRACT_MEAN=SUBTRACT_MEAN, HAS_BIAS=HAS_BIAS,
            BLOCK_R=BLOCK_R, ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        )


@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_PROGRAM_Q": q, "ROWS_PER_PROGRAM_KZ": kv})
        for q, kv in ((1, 2), (1, 4), (1, 8), (1, 16), (2, 2), (2, 4), (2, 8), (2, 16), (2, 32), (4, 16), (4, 32))
    ],
    key=["M_q", "M_k", "head_dim", "rotary_dim"],
)
@triton.jit
def _indexer_qkz_norm_rope_kernel(
    x_q_ptr,
    x_k_ptr,
    x_z_ptr,
    out_q_ptr,
    out_k_ptr,
    out_z_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    M_q,
    M_k,
    stride_x_q,
    stride_x_k,
    stride_x_z,
    stride_out_q,
    stride_out_k,
    stride_cos,
    eps,
    weight_bias_q,
    weight_bias_k,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    SUBTRACT_MEAN_Q: tl.constexpr,
    HAS_BIAS_Q: tl.constexpr,
    SUBTRACT_MEAN_K: tl.constexpr,
    HAS_BIAS_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROWS_PER_PROGRAM_Q: tl.constexpr,
    ROWS_PER_PROGRAM_KZ: tl.constexpr,
    H_CHUNK_Q: tl.constexpr,
    H_CHUNK_K: tl.constexpr,
    N_FULL_Q: tl.constexpr,
    REM_Q: tl.constexpr,
    N_FULL_K: tl.constexpr,
    REM_K: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim
    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim

    w_all_q = tl.load(q_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    w_real_q = tl.load(q_weight_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    w_imag_q = tl.load(q_weight_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    if HAS_BIAS_Q:
        b_all_q = tl.load(q_bias_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
        b_real_q = tl.load(q_bias_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
        b_imag_q = tl.load(q_bias_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    else:
        b_all_q = tl.zeros((BLOCK_D,), dtype=tl.float32)
        b_real_q = tl.zeros((BLOCK_R,), dtype=tl.float32)
        b_imag_q = tl.zeros((BLOCK_R,), dtype=tl.float32)
    num_q_tokens = M_q // num_q_heads
    num_q_tiles = (num_q_tokens + ROWS_PER_PROGRAM_Q - 1) // ROWS_PER_PROGRAM_Q
    for t in range(pid, num_q_tiles, nprog):
        token_start = t * ROWS_PER_PROGRAM_Q
        token_idx = token_start + tl.arange(0, ROWS_PER_PROGRAM_Q)
        token_ok = token_idx < num_q_tokens
        _norm_rope_tile_npu(
            x_q_ptr, out_q_ptr, cos_ptr, sin_ptr, positions_ptr,
            token_idx, token_start, token_ok,
            stride_x_q, stride_out_q, stride_cos, 0,
            w_all_q, w_real_q, w_imag_q, b_all_q, b_real_q, b_imag_q, eps, weight_bias_q,
            num_q_tokens,
            num_heads=num_q_heads, head_dim=head_dim, rotary_dim=rotary_dim,
            SUBTRACT_MEAN=SUBTRACT_MEAN_Q, HAS_BIAS=HAS_BIAS_Q,
            BLOCK_D=BLOCK_D, BLOCK_R=BLOCK_R, ROWS_PER_PROGRAM=ROWS_PER_PROGRAM_Q,
            H_CHUNK=H_CHUNK_Q, N_FULL=N_FULL_Q, REM_HEADS=REM_Q,
        )

    w_all_k = tl.load(k_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    w_real_k = tl.load(k_weight_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    w_imag_k = tl.load(k_weight_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    if HAS_BIAS_K:
        b_all_k = tl.load(k_bias_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
        b_real_k = tl.load(k_bias_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
        b_imag_k = tl.load(k_bias_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    else:
        b_all_k = tl.zeros((BLOCK_D,), dtype=tl.float32)
        b_real_k = tl.zeros((BLOCK_R,), dtype=tl.float32)
        b_imag_k = tl.zeros((BLOCK_R,), dtype=tl.float32)
    num_k_tokens = M_k // num_k_heads
    num_k_tiles = (num_k_tokens + ROWS_PER_PROGRAM_KZ - 1) // ROWS_PER_PROGRAM_KZ
    for t in range(pid, num_k_tiles, nprog):
        token_start = t * ROWS_PER_PROGRAM_KZ
        token_idx = token_start + tl.arange(0, ROWS_PER_PROGRAM_KZ)
        token_ok = token_idx < num_k_tokens
        _norm_rope_tile_npu(
            x_k_ptr, out_k_ptr, cos_ptr, sin_ptr, positions_ptr,
            token_idx, token_start, token_ok,
            stride_x_k, stride_out_k, stride_cos, 0,
            w_all_k, w_real_k, w_imag_k, b_all_k, b_real_k, b_imag_k, eps, weight_bias_k,
            num_k_tokens,
            num_heads=num_k_heads, head_dim=head_dim, rotary_dim=rotary_dim,
            SUBTRACT_MEAN=SUBTRACT_MEAN_K, HAS_BIAS=HAS_BIAS_K,
            BLOCK_D=BLOCK_D, BLOCK_R=BLOCK_R, ROWS_PER_PROGRAM=ROWS_PER_PROGRAM_KZ,
            H_CHUNK=H_CHUNK_K, N_FULL=N_FULL_K, REM_HEADS=REM_K,
        )
        z_bp = tl.make_block_ptr(
            base=x_z_ptr,
            shape=(num_k_tokens, num_k_heads * head_dim),
            strides=(stride_x_z, 1),
            offsets=(token_start, 0),
            block_shape=(ROWS_PER_PROGRAM_KZ, num_k_heads * head_dim),
            order=(1, 0),
        )
        z_vals = tl.load(z_bp, boundary_check=(0,), padding_option="zero")
        z_out_bp = tl.make_block_ptr(
            base=out_z_ptr,
            shape=(num_k_tokens, num_k_heads * head_dim),
            strides=(num_k_heads * head_dim, 1),
            offsets=(token_start, 0),
            block_shape=(ROWS_PER_PROGRAM_KZ, num_k_heads * head_dim),
            order=(1, 0),
        )
        tl.store(z_out_bp, z_vals.to(out_z_ptr.dtype.element_ty), boundary_check=(0,))


def _fused_indexer_qkz_norm_rope_impl(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    q_norm_bias: torch.Tensor | None = None,
    k_norm_bias: torch.Tensor | None = None,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
    k_norm_weight_bias: float = 0.0,
    subtract_mean_q: bool = False,
    subtract_mean_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ``rotary_dim`` follows the module convention: the number of rotated
    # dims at the head of each indexer head (sparse_indexer_rope_dim, e.g.
    # 32). The kernel works in NeoX pairs: real = dims[:rotary_pairs],
    # imag = dims[rotary_pairs:2*rotary_pairs], and indexes the cos/sin cache
    # rows (width = rotary_pairs) by pair id. Passing the dim count through
    # unchanged made the kernel treat 32 dims as 32 pairs -- imag landed on
    # dims 32-63 and cache reads ran past the 16-wide row into the next
    # position's angles. Convert dims -> pairs here.
    rotary_pairs = rotary_dim // 2

    tokens = index_q.shape[0]
    if index_k.shape[0] != tokens or index_z.shape[0] != tokens:
        raise ValueError(
            f"index_q/k/z must have the same number of tokens, got "
            f"{index_q.shape[0]}, {index_k.shape[0]}, {index_z.shape[0]}"
        )
    if index_q.shape[1] != num_q_heads * head_dim:
        raise ValueError(
            f"index_q shape {tuple(index_q.shape)} does not match "
            f"num_q_heads={num_q_heads} * head_dim={head_dim}"
        )
    if index_k.shape[1] != num_k_heads * head_dim or index_z.shape[1] != num_k_heads * head_dim:
        raise ValueError(
            f"index_k and index_z must have shape ({tokens}, {num_k_heads * head_dim}), "
            f"got {tuple(index_k.shape)} and {tuple(index_z.shape)}"
        )
    # index_q comes out pre-quantized to e4m3: its only consumer (the fused
    # indexer scoring) wants fp8, so the quantization rides along with the
    # norm+rope store instead of a separate cast at the scoring entry.
    out_q = torch.empty(
        (tokens, num_q_heads * head_dim),
        device=index_q.device,
        dtype=torch.float8_e4m3fn,
    )
    out_k = torch.empty((tokens, num_k_heads * head_dim), device=index_q.device, dtype=index_k.dtype)
    out_z = torch.empty((tokens, num_k_heads * head_dim), device=index_q.device, dtype=index_z.dtype)

    m_q = tokens * num_q_heads
    m_k = tokens * num_k_heads

    grid = lambda meta: (
        min(
            max(
                (tokens + meta["ROWS_PER_PROGRAM_Q"] - 1) // meta["ROWS_PER_PROGRAM_Q"],
                (tokens + meta["ROWS_PER_PROGRAM_KZ"] - 1) // meta["ROWS_PER_PROGRAM_KZ"],
            ),
            _npu_vec_core_num(),
        ),
    )
    _indexer_qkz_norm_rope_kernel[grid](
        index_q,
        index_k,
        index_z,
        out_q,
        out_k,
        out_z,
        q_norm_weight,
        q_norm_bias if q_norm_bias is not None else q_norm_weight,
        k_norm_weight,
        k_norm_bias if k_norm_bias is not None else k_norm_weight,
        cos,
        sin,
        positions,
        m_q,
        m_k,
        index_q.stride(0),
        index_k.stride(0),
        index_z.stride(0),
        out_q.stride(0),
        out_k.stride(0),
        cos.stride(0),
        eps,
        q_norm_weight_bias,
        k_norm_weight_bias,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        head_dim=head_dim,
        rotary_dim=rotary_pairs,
        SUBTRACT_MEAN_Q=subtract_mean_q,
        HAS_BIAS_Q=q_norm_bias is not None,
        SUBTRACT_MEAN_K=subtract_mean_k,
        HAS_BIAS_K=k_norm_bias is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=triton.next_power_of_2(rotary_pairs),
        H_CHUNK_Q=(hc_q := _pick_head_chunk(num_q_heads, head_dim)),
        H_CHUNK_K=(hc_k := _pick_head_chunk(num_k_heads, head_dim)),
        N_FULL_Q=num_q_heads // hc_q,
        REM_Q=num_q_heads % hc_q,
        N_FULL_K=num_k_heads // hc_k,
        REM_K=num_k_heads % hc_k,
    )
    return out_q, out_k, out_z


def _fused_indexer_qkz_norm_rope_fake(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    q_norm_bias: torch.Tensor | None = None,
    k_norm_bias: torch.Tensor | None = None,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
    k_norm_weight_bias: float = 0.0,
    subtract_mean_q: bool = False,
    subtract_mean_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = index_q.shape[0]
    q = torch.empty(
        (tokens, num_q_heads * head_dim),
        device=index_q.device,
        dtype=torch.float8_e4m3fn,
    )
    k = torch.empty((tokens, num_k_heads * head_dim), device=index_k.device, dtype=index_k.dtype)
    z = torch.empty((tokens, num_k_heads * head_dim), device=index_z.device, dtype=index_z.dtype)
    return q, k, z


direct_register_custom_op(
    op_name="ascend_step5_fused_indexer_norm_rope",
    op_func=_fused_indexer_qkz_norm_rope_impl,
    mutates_args=[],
    fake_impl=_fused_indexer_qkz_norm_rope_fake,
)


def fused_indexer_qkz_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    q_norm_bias: torch.Tensor | None = None,
    k_norm_bias: torch.Tensor | None = None,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
    k_norm_weight_bias: float = 0.0,
    subtract_mean_q: bool = False,
    subtract_mean_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.ascend_step5_fused_indexer_norm_rope(
        index_q,
        index_k,
        index_z,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        positions,
        q_norm_bias,
        k_norm_bias,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=rotary_dim,
        eps=eps,
        q_norm_weight_bias=q_norm_weight_bias,
        k_norm_weight_bias=k_norm_weight_bias,
        subtract_mean_q=subtract_mean_q,
        subtract_mean_k=subtract_mean_k,
    )
