# SPDX-License-Identifier: Apache-2.0
"""indexer_qkz_norm_rope -- Ascend release build (self-contained).

Independent publishable operator: the Step5 sparse-indexer Q/K norm + RoPE
and the Z passthrough fused into one Triton kernel for Ascend (Triton-Ascend
backend). It replaces the eager indexer chain in the DSA layers --
per-head RMSNorm on index_q, per-head LayerNorm on index_k, NeoX RoPE on
both, and a contiguous copy of index_z -- which materialized every
intermediate back to global memory, with one kernel launch that reads the
strided ``split_indexer`` views of the packed qkv+indexer GEMM output in
place (no materializing copy on any of the three inputs).

Numerical contract:

- index_q: RMSNorm per head over ``head_dim`` in fp32, zero-centered weights
  add 1.0 (``q_norm_weight_bias``), no mean subtraction, no bias; rounded to
  the input dtype (bf16) before rotation, rotated in fp32.
- index_k: LayerNorm per head (mean subtracted, learned ``k_norm_bias``),
  same rounding/rotation contract.
- RoPE: NeoX pairs over the first ``rotary_dim`` dims = ``rotary_dim // 2``
  pairs (pair layout: dims i and i + rotary_dim // 2), angles read from the
  a single caller-provided ``[cos | sin]`` cache (``cos_sin_cache``),
  rows indexed by ``positions``.
- index_q is stored as float8_e4m3fn: its only consumer (the fused indexer
  scoring, ``indexer_score_topk``) wants fp8, so the quantization rides
  along with the norm+rope store instead of a separate cast at the scoring
  entry. The
  store rounds through the input precision first (bf16 -> e4m3), which
  reproduces the unfused pipeline's bits up to one e4m3 ULP.
- index_k / index_z keep the input dtype (Z is a bit-exact passthrough).

``rotary_dim`` follows the module convention (the number of rotated dims at
the head of each indexer head, ``sparse_indexer_rope_dim``); the kernel works
in NeoX pairs, so the wrapper converts dims -> pairs. Passing the dim count
through unchanged made the kernel treat 32 dims as 32 pairs -- the
imaginary half landed on dims 32-63 and cache reads ran past the 16-wide row
into the next position's angles (the regression the pos-0 identity test
guards).
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:
    try:
        import triton.language.extra.cann.extension as _extension_module  # type: ignore
    except ImportError:
        _extension_module = None


def _npu_vec_core_num() -> int:
    # Reuse the shared helper. Its initialization is idempotent, so device
    # properties are queried only once per process rather than on every launch.
    from vllm_ascend.ops.triton.triton_utils import (
        get_vectorcore_num,
        init_device_properties_triton,
    )

    init_device_properties_triton()
    return get_vectorcore_num()


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
    LAYERNORM: tl.constexpr,
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
        base=x_ptr,
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

    if LAYERNORM:
        mean = tl.sum(values, axis=1) / head_dim
        centered = values - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / head_dim
    else:
        mean = tl.zeros((ROWS_PER_PROGRAM * CH,), dtype=tl.float32)
        variance = tl.sum(values * values, axis=1) / head_dim
    rstd = tl.rsqrt(variance + eps)
    result_tail = (values - mean[:, None]) * rstd[:, None] * (w_hd + weight_bias)
    if LAYERNORM:
        result_tail = result_tail + b_hd

    if CH == 1:
        xrow = tok_abs * stride_x_token + head_off + head * head_dim
        real_raw = tl.load(
            x_ptr + xrow[:, None] + half[None, :], mask=row_ok[:, None] & half_ok[None, :], other=0.0
        ).to(tl.float32)
        imag_raw = tl.load(
            x_ptr + xrow[:, None] + rotary_dim + half[None, :], mask=row_ok[:, None] & half_ok[None, :], other=0.0
        ).to(tl.float32)
    else:
        real_raw = _extension_module.extract_slice(values, [0, 0], [ROWS_PER_PROGRAM * CH, BLOCK_R], [1, 1])
        imag_raw = _extension_module.extract_slice(values, [0, rotary_dim], [ROWS_PER_PROGRAM * CH, BLOCK_R], [1, 1])
    real = (real_raw - mean[:, None]) * rstd[:, None] * (w_real + weight_bias)
    imaginary = (imag_raw - mean[:, None]) * rstd[:, None] * (w_imag + weight_bias)
    if LAYERNORM:
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
    cos_sin_cache_ptr,
    positions_ptr,
    token_idx,
    token_start,
    token_ok,
    stride_x_token,
    stride_out_token,
    stride_cos,
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
    LAYERNORM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    H_CHUNK: tl.constexpr,
    N_FULL: tl.constexpr,
    REM_HEADS: tl.constexpr,
) -> None:
    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    w_hd = _extension_module.extract_slice(tl.reshape(w_all, (1, BLOCK_D)), [0, 0], [1, head_dim], [1, 1])
    if LAYERNORM:
        b_hd = _extension_module.extract_slice(tl.reshape(b_all, (1, BLOCK_D)), [0, 0], [1, head_dim], [1, 1])
    else:
        b_hd = tl.zeros((1, head_dim), dtype=tl.float32)

    position = tl.load(positions_ptr + token_idx, mask=token_ok, other=0).to(tl.int32)
    cos_t = tl.load(
        cos_sin_cache_ptr + position[:, None] * stride_cos + half[None, :],
        mask=token_ok[:, None] & half_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    _extension_module.compile_hint(cos_t, "mayDiscretememaccess")
    sin_t = tl.load(
        cos_sin_cache_ptr + rotary_dim + position[:, None] * stride_cos + half[None, :],
        mask=token_ok[:, None] & half_ok[None, :],
        other=0.0,
    ).to(tl.float32)
    _extension_module.compile_hint(sin_t, "mayDiscretememaccess")

    for c in tl.static_range(N_FULL):
        _norm_rope_tile_npu_chunk(
            x_ptr,
            out_ptr,
            cos_t,
            sin_t,
            token_start,
            stride_x_token,
            stride_out_token,
            c * H_CHUNK * head_dim,
            w_hd,
            b_hd,
            w_real,
            w_imag,
            b_real,
            b_imag,
            eps,
            weight_bias,
            tokens,
            CH=H_CHUNK,
            TOTAL_HEADS=num_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            LAYERNORM=LAYERNORM,
            BLOCK_R=BLOCK_R,
            ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        )
    if REM_HEADS > 0:
        _norm_rope_tile_npu_chunk(
            x_ptr,
            out_ptr,
            cos_t,
            sin_t,
            token_start,
            stride_x_token,
            stride_out_token,
            N_FULL * H_CHUNK * head_dim,
            w_hd,
            b_hd,
            w_real,
            w_imag,
            b_real,
            b_imag,
            eps,
            weight_bias,
            tokens,
            CH=REM_HEADS,
            TOTAL_HEADS=num_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            LAYERNORM=LAYERNORM,
            BLOCK_R=BLOCK_R,
            ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        )


@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_PROGRAM_Q": q, "ROWS_PER_PROGRAM_KZ": kv})
        for q, kv in ((1, 2), (1, 4), (1, 8), (1, 16), (2, 2), (2, 4), (2, 8), (2, 16), (2, 32), (4, 16), (4, 32))
    ],
    # Token counts are intentionally excluded: tune once per layer shape and
    # reuse the selected configuration for all subsequent token counts.
    key=["head_dim", "rotary_dim"],
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
    k_weight_ptr,
    k_bias_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    tokens,
    stride_x_q,
    stride_x_k,
    stride_x_z,
    stride_out_q,
    stride_out_k,
    stride_cos,
    eps,
    weight_bias_q,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
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
    b_all_q = tl.zeros((BLOCK_D,), dtype=tl.float32)
    b_real_q = tl.zeros((BLOCK_R,), dtype=tl.float32)
    b_imag_q = tl.zeros((BLOCK_R,), dtype=tl.float32)
    num_q_tiles = (tokens + ROWS_PER_PROGRAM_Q - 1) // ROWS_PER_PROGRAM_Q
    for t in range(pid, num_q_tiles, nprog):
        token_start = t * ROWS_PER_PROGRAM_Q
        token_idx = token_start + tl.arange(0, ROWS_PER_PROGRAM_Q)
        token_ok = token_idx < tokens
        _norm_rope_tile_npu(
            x_q_ptr,
            out_q_ptr,
            cos_sin_cache_ptr,
            positions_ptr,
            token_idx,
            token_start,
            token_ok,
            stride_x_q,
            stride_out_q,
            stride_cos,
            w_all_q,
            w_real_q,
            w_imag_q,
            b_all_q,
            b_real_q,
            b_imag_q,
            eps,
            weight_bias_q,
            tokens,
            num_heads=num_q_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            LAYERNORM=False,
            BLOCK_D=BLOCK_D,
            BLOCK_R=BLOCK_R,
            ROWS_PER_PROGRAM=ROWS_PER_PROGRAM_Q,
            H_CHUNK=H_CHUNK_Q,
            N_FULL=N_FULL_Q,
            REM_HEADS=REM_Q,
        )

    w_all_k = tl.load(k_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    w_real_k = tl.load(k_weight_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    w_imag_k = tl.load(k_weight_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    b_all_k = tl.load(k_bias_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    b_real_k = tl.load(k_bias_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    b_imag_k = tl.load(k_bias_ptr + rotary_dim + half, mask=half_ok, other=0.0).to(tl.float32)
    num_k_tiles = (tokens + ROWS_PER_PROGRAM_KZ - 1) // ROWS_PER_PROGRAM_KZ
    for t in range(pid, num_k_tiles, nprog):
        token_start = t * ROWS_PER_PROGRAM_KZ
        token_idx = token_start + tl.arange(0, ROWS_PER_PROGRAM_KZ)
        token_ok = token_idx < tokens
        _norm_rope_tile_npu(
            x_k_ptr,
            out_k_ptr,
            cos_sin_cache_ptr,
            positions_ptr,
            token_idx,
            token_start,
            token_ok,
            stride_x_k,
            stride_out_k,
            stride_cos,
            w_all_k,
            w_real_k,
            w_imag_k,
            b_all_k,
            b_real_k,
            b_imag_k,
            eps,
            0.0,
            tokens,
            num_heads=num_k_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            LAYERNORM=True,
            BLOCK_D=BLOCK_D,
            BLOCK_R=BLOCK_R,
            ROWS_PER_PROGRAM=ROWS_PER_PROGRAM_KZ,
            H_CHUNK=H_CHUNK_K,
            N_FULL=N_FULL_K,
            REM_HEADS=REM_K,
        )
        z_bp = tl.make_block_ptr(
            base=x_z_ptr,
            shape=(tokens, num_k_heads * head_dim),
            strides=(stride_x_z, 1),
            offsets=(token_start, 0),
            block_shape=(ROWS_PER_PROGRAM_KZ, num_k_heads * head_dim),
            order=(1, 0),
        )
        z_vals = tl.load(z_bp, boundary_check=(0,), padding_option="zero")
        z_out_bp = tl.make_block_ptr(
            base=out_z_ptr,
            shape=(tokens, num_k_heads * head_dim),
            strides=(num_k_heads * head_dim, 1),
            offsets=(token_start, 0),
            block_shape=(ROWS_PER_PROGRAM_KZ, num_k_heads * head_dim),
            order=(1, 0),
        )
        tl.store(z_out_bp, z_vals.to(out_z_ptr.dtype.element_ty), boundary_check=(0,))


def fused_indexer_qkz_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    k_norm_bias: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    q_norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Step5 sparse-indexer Q/K norm + RoPE with Z passthrough.

    ``index_q`` is ``[tokens, num_q_heads * head_dim]`` (RMSNorm, zero-
    centered weights add ``q_norm_weight_bias``), ``index_k`` /
    ``index_z`` are ``[tokens, num_k_heads * head_dim]`` (K: LayerNorm with
    ``k_norm_bias``; Z: passthrough). Inputs may be strided views (the
    ``split_indexer`` slices of the packed qkv+indexer GEMM output) -- the
    kernel reads them in place through their row strides. RoPE angles come
    from a single caller-provided ``[cos | sin]`` cache
    (``cos_sin_cache``), rows indexed by ``positions``; ``rotary_dim`` is
    the rotated dim count and is converted to NeoX pairs internally.
    Returns ``(index_q e4m3, index_k bf16, index_z bf16)``: the fp8
    quantization of index_q rides along with the norm+rope store (its only
    consumer, the fused indexer scoring, wants fp8).
    """
    # ``rotary_dim`` follows the module convention: the number of rotated
    # dims at the head of each indexer head (sparse_indexer_rope_dim, e.g.
    # 32). The kernel works in NeoX pairs: real = dims[:rotary_pairs],
    # imag = dims[rotary_pairs:2*rotary_pairs], and indexes the cos/sin cache
    # rows (width = rotary_pairs) by pair id. Passing the dim count through
    # unchanged made the kernel treat 32 dims as 32 pairs -- imag landed on
    # dims 32-63 and cache reads ran past the 16-wide row into the next
    # position's angles. Convert dims -> pairs here.
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if cos_sin_cache.shape[-1] != rotary_dim:
        raise ValueError(
            "cos_sin_cache must be laid out as [cos | sin] with last "
            f"dimension {rotary_dim}, got shape {tuple(cos_sin_cache.shape)}"
        )
    rotary_pairs = rotary_dim // 2

    tokens = index_q.shape[0]
    if index_k.shape[0] != tokens or index_z.shape[0] != tokens:
        raise ValueError(
            f"index_q/k/z must have the same number of tokens, got "
            f"{index_q.shape[0]}, {index_k.shape[0]}, {index_z.shape[0]}"
        )
    if index_q.shape[1] != num_q_heads * head_dim:
        raise ValueError(
            f"index_q shape {tuple(index_q.shape)} does not match num_q_heads={num_q_heads} * head_dim={head_dim}"
        )
    if index_k.shape[1] != num_k_heads * head_dim or index_z.shape[1] != num_k_heads * head_dim:
        raise ValueError(
            f"index_k and index_z must have shape ({tokens}, {num_k_heads * head_dim}), "
            f"got {tuple(index_k.shape)} and {tuple(index_z.shape)}"
        )
    out_q = index_q.new_empty((tokens, num_q_heads * head_dim), dtype=torch.float8_e4m3fn)
    out_k = index_k.new_empty((tokens, num_k_heads * head_dim))
    out_z = index_z.new_empty((tokens, num_k_heads * head_dim))
    if tokens == 0:
        return out_q, out_k, out_z
    # index_q comes out pre-quantized to e4m3: its only consumer (the fused
    # indexer scoring) wants fp8, so the quantization rides along with the
    # norm+rope store instead of a separate cast at the scoring entry.

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
        k_norm_weight,
        k_norm_bias,
        cos_sin_cache,
        positions,
        tokens,
        index_q.stride(0),
        index_k.stride(0),
        index_z.stride(0),
        out_q.stride(0),
        out_k.stride(0),
        cos_sin_cache.stride(0),
        eps,
        q_norm_weight_bias,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        head_dim=head_dim,
        rotary_dim=rotary_pairs,
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
