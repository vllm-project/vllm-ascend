import torch
import torch_npu


# KV cache MXFP8 scale layouts. The block and head axes are ordered the way
# QuantFlashAttn's PA_BBND reads them -- the same order as the K/V caches
# themselves ([num_blocks, block_size, num_kv_heads, head_dim]) -- so attention
# consumes cache and scales without transposing either (validated on-device by
# the vendored-QFA bring-up; the public ops-transformer doc lists PA_BBND
# k_descale (Bn, Bs, KV_N, D/64, 2) / v_descale (Bn, Bs/64, KV_N, D, 2)).
# K scale token:  [num_tokens, num_kv_heads, head_dim // 64, 2]
# K scale cache:  [num_blocks, block_size, num_kv_heads, head_dim // 64, 2]
# V scale token (axis=0 quant): [cdiv(num_tokens, 64), num_kv_heads, head_dim, 2]
# V scale cache:  [num_blocks, block_size // 64, num_kv_heads, head_dim, 2]
MXFP_KV_SCALE_GROUP_SIZE = 64
MXFP_KV_SCALE_VALUES_PER_GROUP = 2
# Unified per-block scale bytes: num_kv_heads * block_size * head_dim / MXFP8_GROUP_SIZE (K and V).
MXFP8_GROUP_SIZE = 32
# E8M0 scale elements are always 1 byte in KV cache budgeting.
MXFP_SCALE_DTYPE_SIZE = 1


def validate_mxfp_k_scale_head_dim(head_dim: int) -> None:
    if head_dim % MXFP_KV_SCALE_GROUP_SIZE != 0:
        raise ValueError(
            f"C8_MXFP K scale cache requires head_dim divisible by {MXFP_KV_SCALE_GROUP_SIZE}, got {head_dim}."
        )


def validate_mxfp_v_scale_block_size(block_size: int) -> None:
    if block_size % MXFP_KV_SCALE_GROUP_SIZE != 0:
        raise ValueError(
            f"C8_MXFP V scale cache requires block_size divisible by {MXFP_KV_SCALE_GROUP_SIZE}, got {block_size}."
        )


def mxfp_kv_scale_groups(head_dim: int) -> int:
    validate_mxfp_k_scale_head_dim(head_dim)
    return head_dim // MXFP_KV_SCALE_GROUP_SIZE


def mxfp_kv_block_scale_groups(block_size: int) -> int:
    validate_mxfp_v_scale_block_size(block_size)
    return block_size // MXFP_KV_SCALE_GROUP_SIZE


def mxfp_k_scale_page_bytes(num_kv_heads: int, block_size: int, head_dim: int) -> int:
    """Bytes per block for k_scale cache."""
    validate_mxfp_k_scale_head_dim(head_dim)
    return num_kv_heads * block_size * head_dim // MXFP8_GROUP_SIZE


def mxfp_v_scale_page_bytes(num_kv_heads: int, block_size: int, head_dim: int) -> int:
    """Bytes per block for v_scale cache."""
    validate_mxfp_v_scale_block_size(block_size)
    return num_kv_heads * block_size * head_dim // MXFP8_GROUP_SIZE


def mxfp_k_scale_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int, int, int]:
    return (
        num_blocks,
        block_size,
        num_kv_heads,
        mxfp_kv_scale_groups(head_dim),
        MXFP_KV_SCALE_VALUES_PER_GROUP,
    )


def mxfp_v_scale_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int, int, int]:
    return (
        num_blocks,
        mxfp_kv_block_scale_groups(block_size),
        num_kv_heads,
        head_dim,
        MXFP_KV_SCALE_VALUES_PER_GROUP,
    )


def mxfp_kv_page_size_bytes(
    block_size: int,
    num_kv_heads: int,
    k_dim: int,
    v_dim: int,
    kv_dtype_size: int,
) -> int:
    """Bytes per KV cache page for C8_MXFP (FP8 K/V tensors + E8M0 scale caches)."""
    kv_bytes = block_size * num_kv_heads * (k_dim + v_dim) * kv_dtype_size
    scale_bytes = (
        mxfp_k_scale_page_bytes(num_kv_heads, block_size, k_dim)
        + mxfp_v_scale_page_bytes(num_kv_heads, block_size, v_dim)
    ) * MXFP_SCALE_DTYPE_SIZE
    return kv_bytes + scale_bytes


def mxfp_resolve_kv_cache_layout(
    *,
    raw_k_numel: int,
    raw_v_numel: int,
    raw_k_scale_numel: int,
    raw_v_scale_numel: int,
    block_size: int,
    num_kv_heads: int,
    k_dim: int,
    v_dim: int,
    layer_name: str = "",
    num_blocks_hint: int | None = None,
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
]:
    """Derive C8_MXFP KV cache shapes from spec dims and allocated raw buffer sizes.

    ``num_blocks`` is derived from the k_scale buffer; ``k_dim``/``v_dim`` come from the caller
    (typically ``KVCacheSpec``). All four raw buffers must match the expected numel.

    Returns (k_shape, v_shape, k_scale_shape, v_scale_shape).
    """
    validate_mxfp_v_scale_block_size(block_size)
    validate_mxfp_k_scale_head_dim(k_dim)
    if v_dim != k_dim:
        validate_mxfp_k_scale_head_dim(v_dim)

    k_scale_per_block = mxfp_k_scale_page_bytes(num_kv_heads, block_size, k_dim)
    v_scale_per_block = mxfp_v_scale_page_bytes(num_kv_heads, block_size, v_dim)
    if raw_k_scale_numel % k_scale_per_block != 0:
        raise ValueError(
            f"C8_MXFP k_scale buffer size mismatch for layer={layer_name}: "
            f"raw_k_scale_numel={raw_k_scale_numel}, k_scale_per_block={k_scale_per_block}, "
            f"k_dim={k_dim}, block_size={block_size}, num_kv_heads={num_kv_heads}."
        )
    num_blocks = raw_k_scale_numel // k_scale_per_block
    if num_blocks <= 0:
        raise ValueError(
            f"C8_MXFP invalid num_blocks={num_blocks} for layer={layer_name}, "
            f"raw_k_scale_numel={raw_k_scale_numel}, k_scale_per_block={k_scale_per_block}."
        )
    if num_blocks_hint is not None and num_blocks != num_blocks_hint:
        raise ValueError(
            f"C8_MXFP num_blocks mismatch for layer={layer_name}: "
            f"from_k_scale={num_blocks}, num_blocks_hint={num_blocks_hint}."
        )

    kv_slot_per_block = block_size * num_kv_heads
    expected_k = num_blocks * kv_slot_per_block * k_dim
    expected_v = num_blocks * kv_slot_per_block * v_dim
    expected_k_scale = num_blocks * k_scale_per_block
    expected_v_scale = num_blocks * v_scale_per_block
    if (
        raw_k_numel != expected_k
        or raw_v_numel != expected_v
        or raw_k_scale_numel != expected_k_scale
        or raw_v_scale_numel != expected_v_scale
    ):
        raise ValueError(
            f"C8_MXFP KV cache buffer layout mismatch for layer={layer_name}: "
            f"num_blocks={num_blocks}, k_dim={k_dim}, v_dim={v_dim}, "
            f"raw_k_numel={raw_k_numel} (expected {expected_k}), "
            f"raw_v_numel={raw_v_numel} (expected {expected_v}), "
            f"raw_k_scale_numel={raw_k_scale_numel} (expected {expected_k_scale}), "
            f"raw_v_scale_numel={raw_v_scale_numel} (expected {expected_v_scale}), "
            f"block_size={block_size}, num_kv_heads={num_kv_heads}."
        )

    k_shape = (num_blocks, block_size, num_kv_heads, k_dim)
    v_shape = (num_blocks, block_size, num_kv_heads, v_dim)
    k_scale_shape = mxfp_k_scale_cache_shape(num_blocks, block_size, num_kv_heads, k_dim)
    v_scale_shape = mxfp_v_scale_cache_shape(num_blocks, block_size, num_kv_heads, v_dim)
    return k_shape, v_shape, k_scale_shape, v_scale_shape


def scatter_mxfp_k_scale_cache(
    key_scale: torch.Tensor,
    key_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter per-token K scales into the paged K-scale cache.

    ``key_scale`` shape: ``[num_tokens, num_kv_heads, head_dim // 64, 2]``
    (any 1-byte dtype; callers pass a uint8 view of the E8M0 scale).
    ``key_scale_cache`` shape (PA_BBND, block before head):
    ``[num_blocks, block_size, num_kv_heads, head_dim // 64, 2]``.
    """
    validate_mxfp_v_scale_block_size(block_size)
    slots = slot_mapping.to(torch.long)
    if slots.numel() == 0:
        return

    # Graph replay keeps the captured token shape and marks padded rows with
    # slot -1. Remapping -1 to a sentinel slot and writing back the cached
    # value is NOT a no-op when a real token also targets that slot in the
    # same batch: the duplicate-index write order would let the padding row
    # clobber the real token's scale. Drop the padded rows instead (v1 is
    # eager-only, so the dynamic shape is acceptable).
    valid_slots = slots >= 0
    if not bool(valid_slots.all()):
        slots = slots[valid_slots]
        key_scale = key_scale[valid_slots]
        if slots.numel() == 0:
            return

    block_ids = slots // block_size
    block_offsets = slots % block_size
    key_scale_cache[block_ids, block_offsets] = key_scale


def scatter_mxfp_v_cache(
    quant_value: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter per-token quantized V into the paged V cache.

    ``quant_value`` shape: ``[num_tokens, num_kv_heads, v_dim]``.
    ``value_cache`` shape: ``[num_blocks, block_size, num_kv_heads, v_dim]``.
    """
    validate_mxfp_v_scale_block_size(block_size)
    slots = slot_mapping.to(torch.long)
    if slots.numel() == 0:
        return

    num_kv_heads = quant_value.shape[1]
    v_dim = quant_value.shape[2]
    flat_cache = value_cache.view(-1, num_kv_heads * v_dim)
    torch_npu.npu_scatter_nd_update_(
        flat_cache,
        slots.view(-1, 1),
        quant_value.reshape(quant_value.shape[0], num_kv_heads * v_dim),
    )


def scatter_mxfp_v_scale_cache(
    value_scale: torch.Tensor,
    value_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter per-64-token-group V scales into the paged V-scale cache.

    ``value_scale`` comes from ``npu_dynamic_mx_quant(..., axis=0)`` and has shape
    ``[ceil(num_tokens / 64), num_kv_heads, head_dim, 2]``. The cache layout is
    ``[num_blocks, block_size // 64, num_kv_heads, head_dim, 2]`` (PA_BBND).

    Unused while V's scale is the checkpoint's static per-channel one (a
    static V scale is broadcast into the cache once and never scattered);
    kept for a dynamic-V design. Indexing follows the PA_BBND order the rest
    of this module uses, so it stays correct if a dynamic-V path ever calls it.
    """
    validate_mxfp_v_scale_block_size(block_size)
    num_scales = value_scale.shape[0]
    v_scale_slot_mapping = (slot_mapping // MXFP_KV_SCALE_GROUP_SIZE).unique()
    if v_scale_slot_mapping.numel() != num_scales:
        raise ValueError(
            f"C8_MXFP V scale slot mapping mismatch: expected {v_scale_slot_mapping.numel()}, got {num_scales}."
        )

    v_scale_cache_block_size = mxfp_kv_block_scale_groups(block_size)
    block_ids = v_scale_slot_mapping // v_scale_cache_block_size
    v_scale_cache_offsets = v_scale_slot_mapping % v_scale_cache_block_size
    value_scale_cache[block_ids, v_scale_cache_offsets] = value_scale
