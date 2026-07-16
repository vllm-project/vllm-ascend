# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for MiniMax M3 block-sparse GQA attention on Ascend.

Migrated from reference/vllm_cp/vllm/models/minimax_m3/common/ops/index_topk.py
and sparse_attn.py. The kernels accept K/V cache tensors directly so Ascend's
split cache layout does not need to be materialized into the GPU
``[num_blocks, 2, ...]`` layout.
"""

from __future__ import annotations

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128

TOPK_SELECTION_TILE = 128
TOPK_COMPUTE_MIN_TILE = 16
TOPK_NUM_WARPS = 4
TOPK_NUM_STAGES = 2


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _split_triton_main_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) < 2:
            raise ValueError("Main kv cache tuple must contain K and V tensors")
        k_cache, v_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.ndim != 5:
            raise ValueError(f"Unexpected main kv cache ndim: {kv_cache.ndim}")
        if kv_cache.shape[0] == 2:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        elif kv_cache.shape[1] == 2:
            k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
        else:
            raise ValueError(f"Unexpected main kv cache shape: {tuple(kv_cache.shape)}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(f"Unexpected split main kv cache shapes: {tuple(k_cache.shape)}, {tuple(v_cache.shape)}")
    return k_cache, v_cache


def _is_arch_support_pdl() -> bool:
    if current_platform.device_name == "npu":
        return False
    is_supported = getattr(current_platform, "is_arch_support_pdl", None)
    return bool(is_supported()) if callable(is_supported) else False



_SPARSE_ATTN_NUM_STAGES_KWARG: dict | None = None


def _sparse_attn_num_stages_kwarg() -> dict:
    """Triton ``num_stages`` override for the sparse-attn GEMM kernels.

    Forced only where required: CDNA3 (gfx942) caps LDS at
    64 KB, and the default 2-stage pipeline double-buffers the 128x128 K/V tiles
    to ~66 KB ("out of resource: shared memory"), so pin gfx942 to a single
    stage (~32 KB, which fits). Everywhere else (NVIDIA, CDNA4 gfx950) return an
    empty kwarg and let Triton keep its own default -- don't second-guess it.
    Cached: the arch is fixed per process.
    """
    global _SPARSE_ATTN_NUM_STAGES_KWARG
    if _SPARSE_ATTN_NUM_STAGES_KWARG is None:
        kwarg: dict = {}
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx942

            if on_gfx942():
                kwarg = {"num_stages": 1}
        _SPARSE_ATTN_NUM_STAGES_KWARG = kwarg
    return _SPARSE_ATTN_NUM_STAGES_KWARG


# ---------------------------------------------------------------------------
# Decode kernels (split-K). Decode batches are flattened request-major, with a
# runtime query length used to map each query token back to its request metadata.
# This parallelizes over the selected top-k blocks, producing partials that the
# merge kernel combines (flash-decoding). All chunk counts depend only on shape
# constants so the grid is fixed within a cuda graph. Base-2 (exp2/log2)
# softmax matches the prefill kernel.
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(16, triton.next_power_of_2(args["gqa_group_size"])),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    k_cache_ptr,  # [num_blocks, 128, num_kv_heads, head_dim]
    v_cache_ptr,  # [num_blocks, 128, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    max_blocks,
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_k_blk,
    stride_k_pos,
    stride_k_h,
    stride_k_d,
    stride_v_blk,
    stride_v_pos,
    stride_v_h,
    stride_v_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    USE_PDL: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_valid_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx = tl.load(
        idx_base + off_t * stride_tk,
        mask=off_t < max_topk,
        other=-1,
    )
    topk_valid_blk = (topk_idx >= 0) & (topk_idx < num_valid_blocks) & (topk_idx < max_blocks)
    valid_topk = (off_t < max_topk) & topk_valid_blk
    # Keep all positions up to the last valid selected block. This skips only
    # a trailing invalid suffix; the hot loop still has PR33 per-block guards,
    # so interleaved invalid entries remain precision-safe.
    real_topk = tl.max(tl.where(valid_topk, off_t + 1, 0), axis=0)
    chunk_size_topk = tl.maximum(
        1,
        (real_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS,
    )
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_topk = tl.minimum(chunk_start_topk + chunk_size_topk, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    if chunk_start_topk >= chunk_end_topk:
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
            shape=(gqa_group_size,),
            strides=(stride_l_h,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_H,),
            order=(0,),
        )
        empty_lse = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        tl.store(lse_ptrs, empty_lse, boundary_check=(0,))
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        return

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr += stride_tk
        valid_blk = (blk >= 0) & (blk < num_valid_blocks) & (blk < max_blocks)
        safe_blk = tl.minimum(tl.maximum(blk, 0), max_blocks - 1)
        c = safe_blk * BLOCK_SIZE_K
        page = tl.load(bt_row + safe_blk, mask=valid_blk, other=-1).to(tl.int64)
        valid_page = valid_blk & (page >= 0)
        safe_page = tl.maximum(page, 0)
        pos = c + off_n
        pos_mask = (pos < kv_len) & valid_page
        k = tl.load(
            k_cache_ptr
            + safe_page * stride_k_blk
            + off_n[None, :] * stride_k_pos
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=d_mask[:, None],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        active_i = m_ij > float("-inf")
        p = tl.exp2(qk - m_ij[:, None])
        p = tl.where(active_i[:, None], p, 0.0)
        l_ij = tl.sum(p, axis=1)
        acc_scale = tl.where(active_i, tl.exp2(m_i - m_ij), tl.zeros_like(m_i))
        acc_o = acc_o * acc_scale[:, None]
        v = tl.load(
            v_cache_ptr
            + safe_page * stride_v_blk
            + off_n[:, None] * stride_v_pos
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=d_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_next = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        lse_i = tl.where(active_i, lse_next, lse_i)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])})
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    # NOTE: assume seq_lens is safe to load before gdc_wait()
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    valid_chunk = lse > float("-inf")
    o = tl.load(
        o_ptr + off_c[:, None] * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h + off_d[None, :] * stride_o_d,
        mask=valid_chunk[:, None] & (off_d[None, :] < head_dim),
        other=0.0,
    )
    lse_max = tl.max(lse, axis=0)
    has_lse = lse_max > float("-inf")
    safe_lse_max = tl.where(has_lse, lse_max, 0.0)
    weights = tl.where(lse > float("-inf"), tl.exp2(lse - safe_lse_max), 0.0)
    denom = tl.sum(weights, axis=0)
    denom_safe = tl.where(denom > 0.0, denom, 1.0)
    o_merged = tl.sum(o * weights[:, None], axis=0) / denom_safe
    o_merged = tl.where(has_lse, o_merged, tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32))
    out_ptrs = out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)



@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    k_cache, v_cache = _split_triton_main_kv_cache(kv_cache)
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = k_cache.dtype in _FP8_DTYPES or v_cache.dtype in _FP8_DTYPES
    use_pdl = _is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    # split-K over the selected blocks; keep enough programs for small decode
    # batches without regressing long-context sparse decode.
    TARGET_GRID = 64
    MAX_TOPK_CHUNKS = 4
    target = max(
        1,
        min(
            max_topk,
            MAX_TOPK_CHUNKS,
            TARGET_GRID // max(1, total_q * num_kv_heads),
        ),
    )
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        num_topk_chunks,
        total_q,
        num_heads,
        head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    lse_partial = torch.empty(num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device)
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        block_table.shape[-1],
        total_q,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        USE_PDL=use_pdl,
        **_sparse_attn_num_stages_kwarg(),
        **pdl_launch,
    )

    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
