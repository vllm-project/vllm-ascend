# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 SWA RMSNorm, partial RoPE and paged-cache store.

The projection GEMM remains separate.  One Triton program consumes one WKV
row, normalizes it, applies the model's interleaved partial RoPE, and writes
the result directly to the builder-prepared physical SWA cache coordinate.
No normalized/rotated KV tensor is materialized.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _dsv41_rmsnorm_rope_cache_kernel(
    projected_kv,
    norm_weight,
    cos,
    sin,
    cache,
    slot_mapping,
    KV_TOKEN_STRIDE: tl.constexpr,
    COS_TOKEN_STRIDE: tl.constexpr,
    COS_DIM_STRIDE: tl.constexpr,
    SIN_TOKEN_STRIDE: tl.constexpr,
    SIN_DIM_STRIDE: tl.constexpr,
    SLOT_TOKEN_STRIDE: tl.constexpr,
    SLOT_DIM_STRIDE: tl.constexpr,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    CACHE_DIM_STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, HEAD_DIM)
    row = projected_kv + token * KV_TOKEN_STRIDE

    value = tl.load(row + dims).to(tl.float32)
    weight = tl.load(norm_weight + dims).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(value * value, axis=0) / HEAD_DIM + EPS)

    # The standalone RMSNorm publishes BF16 before RoPE consumes it.  Keep
    # that rounding boundary instead of fusing the two FP32 expressions.
    normalized = (value * inverse_rms * weight).to(projected_kv.dtype.element_ty).to(tl.float32)

    rotary_index = dims - NOPE_DIM
    rotary_mask = dims >= NOPE_DIM
    even = (rotary_index % 2) == 0
    pair_dims = tl.where(rotary_mask, dims + tl.where(even, 1, -1), 0)
    pair_value = tl.load(row + pair_dims, mask=rotary_mask, other=0.0).to(tl.float32)
    pair_weight = tl.load(norm_weight + pair_dims, mask=rotary_mask, other=0.0).to(tl.float32)
    pair_normalized = (pair_value * inverse_rms * pair_weight).to(projected_kv.dtype.element_ty).to(tl.float32)

    cos_value = tl.load(
        cos + token * COS_TOKEN_STRIDE + rotary_index * COS_DIM_STRIDE,
        mask=rotary_mask,
        other=1.0,
    ).to(tl.float32)
    sin_value = tl.load(
        sin + token * SIN_TOKEN_STRIDE + rotary_index * SIN_DIM_STRIDE,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    sin_sign = tl.where(even, -1.0, 1.0)
    rotated = normalized * cos_value + pair_normalized * sin_value * sin_sign
    output = tl.where(rotary_mask, rotated, normalized)

    slot_row = slot_mapping + token * SLOT_TOKEN_STRIDE
    block = tl.load(slot_row).to(tl.int64)
    offset = tl.load(slot_row + SLOT_DIM_STRIDE).to(tl.int64)
    valid = (block >= 0) & (offset >= 0)
    cache_offsets = block * CACHE_BLOCK_STRIDE + offset * CACHE_TOKEN_STRIDE + dims * CACHE_DIM_STRIDE
    tl.store(cache + cache_offsets, output, mask=valid)


def dsv41_rmsnorm_rope_cache(
    projected_kv: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    eps: float,
    nope_dim: int,
) -> None:
    """Normalize/rotate projected KV and write it directly to the SWA cache.

    ``slot_mapping`` contains physical ``[block, offset]`` coordinates.  A
    negative coordinate suppresses the whole row, including graph-padding
    rows.  Cache offsets use the tensor's real strides, so a cache plane may
    remain a view into a layer-outermost allocation.
    """
    assert projected_kv.ndim == 2 and projected_kv.stride(1) == 1
    num_tokens, head_dim = projected_kv.shape
    rope_dim = head_dim - nope_dim
    assert head_dim == 512
    assert 0 <= nope_dim < head_dim and rope_dim % 2 == 0
    assert projected_kv.dtype == torch.bfloat16
    assert norm_weight.device == projected_kv.device
    assert norm_weight.shape == (head_dim,) and norm_weight.is_contiguous()
    assert cos.shape[0] >= num_tokens and sin.shape[0] >= num_tokens
    assert cos.shape[-1] == sin.shape[-1] == rope_dim
    assert cos.device == sin.device == projected_kv.device
    assert cos.stride(-1) == sin.stride(-1) == 1
    assert cache.ndim == 4 and cache.shape[-2:] == (1, head_dim)
    assert cache.device == projected_kv.device
    assert cache.dtype == projected_kv.dtype and cache.stride(-1) == 1
    assert slot_mapping.ndim == 2 and slot_mapping.shape[0] >= num_tokens and slot_mapping.shape[1] == 2
    assert slot_mapping.device == projected_kv.device
    assert slot_mapping.dtype in (torch.int32, torch.int64)
    assert slot_mapping.stride(1) == 1
    if num_tokens == 0:
        return

    cache_rows = cache.squeeze(-2)
    _dsv41_rmsnorm_rope_cache_kernel[(num_tokens,)](
        projected_kv,
        norm_weight,
        cos,
        sin,
        cache_rows,
        slot_mapping,
        projected_kv.stride(0),
        cos.stride(0),
        cos.stride(-1),
        sin.stride(0),
        sin.stride(-1),
        slot_mapping.stride(0),
        slot_mapping.stride(1),
        *cache_rows.stride(),
        head_dim,
        nope_dim,
        eps,
        enable_fp_fusion=False,
    )
