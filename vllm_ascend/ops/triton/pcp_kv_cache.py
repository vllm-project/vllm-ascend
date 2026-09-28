"""PCP KV cache transfer helpers."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes, get_vectorcore_num, init_device_properties_triton


@triton.jit
def _copy_pcp_kv_cache_kernel(
    key_cache,
    rope_cache,
    slots,
    packed,
    num_tokens,
    cache_block_size: tl.constexpr,
    k_stride_block,
    k_stride_offset,
    k_stride_d,
    rope_stride_block,
    rope_stride_offset,
    rope_stride_d,
    k_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    if BLOCK_ROWS == 1:
        # Keep scalar slot addressing for small batches: the Ascend compiler
        # cannot lower the modulo expression in a singleton 2D tile.
        feature_indices = tl.arange(0, BLOCK_COLS)
        for token_idx in range(tl.program_id(0), num_tokens, tl.num_programs(0)):
            slot_idx = tl.load(slots + token_idx).to(tl.int64)
            valid_slot = slot_idx >= 0
            key_offsets = (
                (slot_idx // cache_block_size) * k_stride_block
                + (slot_idx % cache_block_size) * k_stride_offset
                + feature_indices * k_stride_d
            )
            key_values = tl.load(key_cache + key_offsets, mask=valid_slot & (feature_indices < k_dim), other=0)
            tl.store(
                packed + token_idx * (k_dim + rope_dim) + feature_indices,
                key_values,
                mask=feature_indices < k_dim,
            )
            if rope_dim > 0:
                rope_offsets = (
                    (slot_idx // cache_block_size) * rope_stride_block
                    + (slot_idx % cache_block_size) * rope_stride_offset
                    + feature_indices * rope_stride_d
                )
                rope_values = tl.load(
                    rope_cache + rope_offsets, mask=valid_slot & (feature_indices < rope_dim), other=0
                )
                tl.store(
                    packed + token_idx * (k_dim + rope_dim) + k_dim + feature_indices,
                    rope_values,
                    mask=feature_indices < rope_dim,
                )
    else:
        feature_indices = tl.arange(0, BLOCK_COLS)[None, :]
        for tile_idx in range(tl.program_id(0), tl.cdiv(num_tokens, BLOCK_ROWS), tl.num_programs(0)):
            token_indices = tile_idx * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
            slot_indices = tl.load(slots + token_indices, mask=token_indices < num_tokens, other=-1).to(tl.int64)
            valid_slots = (token_indices < num_tokens) & (slot_indices >= 0)
            key_offsets = (
                (slot_indices[:, None] // cache_block_size) * k_stride_block
                + (slot_indices[:, None] % cache_block_size) * k_stride_offset
                + feature_indices * k_stride_d
            )
            key_values = tl.load(
                key_cache + key_offsets, mask=valid_slots[:, None] & (feature_indices < k_dim), other=0
            )
            tl.store(
                packed + token_indices[:, None] * (k_dim + rope_dim) + feature_indices,
                key_values,
                mask=(token_indices[:, None] < num_tokens) & (feature_indices < k_dim),
            )  # type: ignore[index]
            if rope_dim > 0:
                rope_offsets = (
                    (slot_indices[:, None] // cache_block_size) * rope_stride_block
                    + (slot_indices[:, None] % cache_block_size) * rope_stride_offset
                    + feature_indices * rope_stride_d
                )
                rope_values = tl.load(
                    rope_cache + rope_offsets, mask=valid_slots[:, None] & (feature_indices < rope_dim), other=0
                )
                tl.store(
                    packed + token_indices[:, None] * (k_dim + rope_dim) + k_dim + feature_indices,
                    rope_values,
                    mask=(token_indices[:, None] < num_tokens) & (feature_indices < rope_dim),
                )  # type: ignore[index]


def _get_pcp_kv_cache_rows(
    num_tokens: int,
    num_vector_cores: int,
    tile_columns: int,
    element_size_bytes: int,
    num_cache_tensors: int,
) -> int:
    """Bound row batching by core occupancy and a conservative UB estimate."""
    # Reserve half the UB for compiler temporaries and buffering.
    # Per cache, budget two INT64 address tiles plus four payload/scratch tiles.
    # INT8 loads may use FP16 temporaries, so budget at least two bytes per element.
    # This estimates compiler usage; it is not an exact peak-liveness calculation.
    estimated_bytes_per_row = tile_columns * num_cache_tensors * (2 * 8 + 4 * max(element_size_bytes, 2))
    max_rows_by_ub = (get_ub_size_bytes() // 2) // estimated_bytes_per_row
    # Keep the validated maximum, and preserve the single-row path when even one
    # row exceeds the estimate. Wider layouts still need compilation validation.
    max_rows_per_program = max(1, min(8, num_tokens // num_vector_cores, max_rows_by_ub))
    return 1 << (max_rows_per_program.bit_length() - 1)


def copy_pcp_kv_cache(cache, slots):
    """Read selected cache rows into a contiguous tensor, zero-filling -1 slots.

    A single tensor represents the complete C8 row, including RoPE and scales.
    Access that layout through an int8 view to preserve all bits, even when its
    storage dtype is FP8. Two tensors represent separate latent and RoPE caches.
    """
    assert len(cache) in (1, 2)
    k = cache[0]
    assert k.ndim == 4 and k.shape[2] == 1
    if len(cache) == 1:
        assert k.element_size() == 1
        k = k.view(torch.int8)
        r = k  # Unused pointer: rope_dim=0 eliminates the second cache's accesses.
        rope_dim = 0
    else:
        r = cache[1]
        assert r.ndim == 4 and r.shape[2] == 1
        assert k.dtype == r.dtype and k.shape[:3] == r.shape[:3]
        rope_dim = r.shape[-1]
    slots = slots.contiguous()
    packed = torch.empty((slots.numel(), k.shape[-1] + rope_dim), dtype=k.dtype, device=k.device)
    if slots.numel():
        init_device_properties_triton()
        num_cores = get_vectorcore_num()
        block_cols = triton.next_power_of_2(max(k.shape[-1], rope_dim))
        rows = _get_pcp_kv_cache_rows(slots.numel(), num_cores, block_cols, k.element_size(), len(cache))
        _copy_pcp_kv_cache_kernel[(min(triton.cdiv(slots.numel(), rows), num_cores),)](
            k,
            r,
            slots,
            packed,
            slots.numel(),
            k.shape[1],
            k.stride(0),
            k.stride(1),
            k.stride(3),
            r.stride(0),
            r.stride(1),
            r.stride(3),
            k.shape[-1],
            rope_dim,
            block_cols,
            rows,
        )
    return packed
