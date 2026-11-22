# SPDX-License-Identifier: Apache-2.0
"""Tree-only Q/K normalization and dense QKV preparation from convolution output."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_tree_qkv_kernel(
    mixed_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    eps,
    T,
    NK,
    NV,
    DK: tl.constexpr,
    DV: tl.constexpr,
    MIX_ROW_STRIDE,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    norm_programs = tl.cdiv(T * NK, BLOCK_M)
    qk_col = tl.arange(0, BLOCK_K)[None, :]
    if pid < 2 * norm_programs:
        which = pid // norm_programs
        qk_row = (pid % norm_programs) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        xmask = qk_row < T * NK
        qk_mask = xmask & (qk_col < DK)
        qk_src = (qk_row // NK) * MIX_ROW_STRIDE + which * (NK * DK) + (qk_row % NK) * DK + qk_col
        xs = tl.load(mixed_ptr + qk_src, qk_mask, other=0.0).to(tl.float32)
        square = tl.broadcast_to(xs * xs, [BLOCK_M, BLOCK_K])
        square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)
        if which == 0:
            tl.store(q_ptr + qk_col + DK * qk_row, xs * rsqrt, qk_mask)
        else:
            tl.store(k_ptr + qk_col + DK * qk_row, xs * rsqrt, qk_mask)
    else:
        # Disjoint programs copy V exactly once; Q/K programs never write V.
        v_col = tl.arange(0, BLOCK_V)[None, :]
        v_row = (pid - 2 * norm_programs) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        v_mask = (v_row < T * NV) & (v_col < DV)
        v_src = (v_row // NV) * MIX_ROW_STRIDE + 2 * NK * DK + (v_row % NV) * DV + v_col
        value = tl.load(mixed_ptr + v_src, v_mask, other=0.0)
        tl.store(v_ptr + v_row * DV + v_col, value, v_mask)


def prepare_tree_qkv(
    mixed_qkv: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    head_key_dim: int,
    head_value_dim: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized Q/K and dense V, each shaped ``[1, tokens, heads, dim]``.

    This matches the default FLA ``l2norm_fwd_kernel2`` reduction and bf16
    output rounding. The caller must retain the ordinary path when the
    optional FLA norm implementation is selected. Q/K copy and normalization
    are fused; V is materialized in the same launch for the existing dense
    recurrent kernels. All allocations/shapes are host metadata only.
    """
    width = 2 * num_key_heads * head_key_dim + num_value_heads * head_value_dim
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] != width or mixed_qkv.dtype != torch.bfloat16:
        raise ValueError("tree QKV preparation requires bf16 [tokens, 2*NK*DK + NV*DV]")
    if mixed_qkv.stride(1) != 1 or mixed_qkv.stride(0) < width:
        raise ValueError("mixed QKV rows must be dense and non-overlapping")
    tokens = mixed_qkv.shape[0]
    query = torch.empty((1, tokens, num_key_heads, head_key_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    key = torch.empty_like(query)
    value = torch.empty((1, tokens, num_value_heads, head_value_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    if tokens == 0:
        return query, key, value
    block_m = 32  # Keep the upstream default normalization reduction layout.
    grid = (2 * triton.cdiv(tokens * num_key_heads, block_m) + triton.cdiv(tokens * num_value_heads, block_m),)
    _prepare_tree_qkv_kernel[grid](
        mixed_qkv,
        query,
        key,
        value,
        eps,
        tokens,
        num_key_heads,
        num_value_heads,
        head_key_dim,
        head_value_dim,
        mixed_qkv.stride(0),
        BLOCK_K=triton.next_power_of_2(head_key_dim),
        BLOCK_V=triton.next_power_of_2(head_value_dim),
        BLOCK_M=block_m,
    )
    return query, key, value
