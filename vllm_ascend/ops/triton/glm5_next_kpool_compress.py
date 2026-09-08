# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fast path for GLM5 Next KPool compression and cache write."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

TRITON_MAX_BLOCK_D = 128


@triton.jit
def _glm5_next_kpool_compress_kernel(
    kv_cache_ptr,
    slot_k_ptr,
    slot_score_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    compressed_ptr,
    slot_k_stride_n: tl.constexpr,
    slot_k_stride_p: tl.constexpr,
    slot_k_stride_d: tl.constexpr,
    slot_score_stride_n: tl.constexpr,
    slot_score_stride_p: tl.constexpr,
    slot_score_stride_d: tl.constexpr,
    ape_stride_p: tl.constexpr,
    ape_stride_d: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_d: tl.constexpr,
    compressed_stride_n: tl.constexpr,
    compressed_stride_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    cache_num_slots: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    RETURN_COMPRESSED: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
):
    pool_idx = tl.program_id(0)
    dim_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    pool_offsets = tl.arange(0, BLOCK_P)
    dim_mask = dim_offsets < HEAD_DIM
    pool_mask = pool_offsets < POOL_SIZE
    value_mask = pool_mask[:, None] & dim_mask[None, :]

    score_offsets = (
        slot_score_ptr
        + pool_idx * slot_score_stride_n
        + pool_offsets[:, None] * slot_score_stride_p
        + dim_offsets[None, :] * slot_score_stride_d
    )
    ape_offsets = ape_ptr + pool_offsets[:, None] * ape_stride_p + dim_offsets[None, :] * ape_stride_d
    scores = tl.load(score_offsets, mask=value_mask, other=float("-inf")).to(tl.float32)
    scores += tl.load(ape_offsets, mask=value_mask, other=0.0).to(tl.float32)
    scores = tl.where(dim_mask[None, :], scores, 0.0)

    score_max = tl.max(scores, axis=0)
    weights = tl.exp(scores - score_max[None, :])
    weights = weights / tl.sum(weights, axis=0)[None, :]

    k_offsets = (
        slot_k_ptr
        + pool_idx * slot_k_stride_n
        + pool_offsets[:, None] * slot_k_stride_p
        + dim_offsets[None, :] * slot_k_stride_d
    )
    slot_k = tl.load(k_offsets, mask=value_mask, other=0.0).to(tl.float32)
    compressed = tl.sum(weights * slot_k, axis=0)

    if RETURN_COMPRESSED:
        tl.store(
            compressed_ptr + pool_idx * compressed_stride_n + dim_offsets * compressed_stride_d,
            compressed,
            mask=dim_mask,
        )

    if WRITE_CACHE:
        should_write = True
        if HAS_WRITE_MASK:
            should_write = tl.load(write_mask_ptr + pool_idx) != 0

        flat_slot = tl.load(loc_ptr + pool_idx).to(tl.int64)
        valid_slot = (flat_slot >= 0) & (flat_slot < cache_num_slots)
        # Full ACL graphs retain padded rows whose slot mapping is -1.
        # Keep their pointer arithmetic in bounds even though the store is masked.
        safe_slot = tl.where(valid_slot, flat_slot, 0)
        block_id = safe_slot // cache_block_size
        block_offset = safe_slot % cache_block_size
        tl.store(
            kv_cache_ptr
            + block_id * cache_stride_block
            + block_offset * cache_stride_offset
            + dim_offsets * cache_stride_d,
            compressed,
            mask=dim_mask & should_write & valid_slot,
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_inputs(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None,
    return_compressed: bool,
    write_cache: bool,
) -> None:
    if slot_k.ndim != 3:
        raise ValueError(f"slot_k must be [N,P,D], got {slot_k.shape}.")
    if slot_k.dtype != torch.bfloat16:
        raise TypeError(f"slot_k must be bfloat16, got {slot_k.dtype}.")
    if slot_k.shape[1] <= 0 or slot_k.shape[2] <= 0:
        raise ValueError(f"slot_k must have positive pool/head dims, got {slot_k.shape}.")
    if slot_score.shape != slot_k.shape:
        raise ValueError(f"slot_score must match slot_k, got {slot_score.shape} and {slot_k.shape}.")
    if slot_score.dtype != torch.bfloat16:
        raise TypeError(f"slot_score must be bfloat16, got {slot_score.dtype}.")
    if ape.shape != slot_k.shape[1:]:
        raise ValueError(f"ape must be [P,D]={slot_k.shape[1:]}, got {ape.shape}.")
    if ape.dtype != torch.float32:
        raise TypeError(f"ape must be float32, got {ape.dtype}.")
    if loc.shape != (slot_k.shape[0],):
        raise ValueError(f"loc must have shape {(slot_k.shape[0],)}, got {loc.shape}.")
    if loc.dtype != torch.int64:
        raise TypeError(f"loc must be int64, got {loc.dtype}.")
    if kv_cache.ndim != 4 or kv_cache.shape[2:] != (1, slot_k.shape[2]):
        raise ValueError(f"kv_cache must be [blocks,block,1,{slot_k.shape[2]}], got {kv_cache.shape}.")
    if kv_cache.dtype != torch.bfloat16:
        raise TypeError(f"kv_cache must be bfloat16, got {kv_cache.dtype}.")
    if write_mask is not None:
        if write_mask.shape != (slot_k.shape[0],):
            raise ValueError(f"write_mask must have shape {(slot_k.shape[0],)}, got {write_mask.shape}.")
        if return_compressed:
            raise ValueError("return_compressed cannot be combined with write_mask.")
    if not write_cache and not return_compressed:
        raise ValueError("At least one output must be requested.")
    for name, tensor in (
        ("kv_cache", kv_cache),
        ("slot_score", slot_score),
        ("ape", ape),
        ("loc", loc),
    ):
        if tensor.device != slot_k.device:
            raise ValueError(f"{name} must be on {slot_k.device}, got {tensor.device}.")
    if write_mask is not None and write_mask.device != slot_k.device:
        raise ValueError(f"write_mask must be on {slot_k.device}, got {write_mask.device}.")


def glm5_next_kpool_compress_and_write_cache_triton(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    *,
    write_mask: torch.Tensor | None = None,
    return_compressed: bool = False,
    write_cache: bool = True,
) -> torch.Tensor | None:
    """Compress KPool keys and optionally write them into the paged BF16 cache."""
    _validate_inputs(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        return_compressed,
        write_cache,
    )

    if not slot_k.is_contiguous():
        slot_k = slot_k.contiguous()
    if not slot_score.is_contiguous():
        slot_score = slot_score.contiguous()
    if not ape.is_contiguous():
        ape = ape.contiguous()
    if not loc.is_contiguous():
        loc = loc.contiguous()
    if write_mask is not None and not write_mask.is_contiguous():
        write_mask = write_mask.contiguous()

    num_pools, pool_size, head_dim = slot_k.shape
    if num_pools == 0:
        if return_compressed:
            return torch.empty((0, head_dim), dtype=torch.bfloat16, device=slot_k.device)
        return None

    compressed = None
    compressed_ptr = kv_cache
    compressed_stride_n = 0
    compressed_stride_d = 0
    if return_compressed:
        compressed = torch.empty((num_pools, head_dim), dtype=torch.bfloat16, device=slot_k.device)
        compressed_ptr = compressed
        compressed_stride_n = compressed.stride(0)
        compressed_stride_d = compressed.stride(1)

    block_p = _next_power_of_2(pool_size)
    block_d = min(_next_power_of_2(head_dim), TRITON_MAX_BLOCK_D)
    write_mask_arg = write_mask if write_mask is not None else loc
    _glm5_next_kpool_compress_kernel[(num_pools, triton.cdiv(head_dim, block_d))](
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask_arg,
        compressed_ptr,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_k.stride(2),
        slot_score.stride(0),
        slot_score.stride(1),
        slot_score.stride(2),
        ape.stride(0),
        ape.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(3),
        compressed_stride_n,
        compressed_stride_d,
        kv_cache.shape[1],
        kv_cache.shape[0] * kv_cache.shape[1],
        pool_size,
        head_dim,
        block_p,
        block_d,
        write_cache,
        return_compressed,
        write_mask is not None,
    )
    return compressed
