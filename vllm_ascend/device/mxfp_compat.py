import torch
import torch_npu

# TODO(linfeng): Temporary compatibility shim for MXFP4/MXFP8 because current torch_npu
# releases do not expose the required dtype attributes yet. Simplify or remove this
# file after the torch_npu release in March 2026 includes those dtype symbols.
FLOAT8_E8M0FNU_DTYPE = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))
FLOAT4_E2M1FN_X2_DTYPE = getattr(torch_npu, "float4_e2m1fn_x2", getattr(torch, "float4_e2m1fn_x2", None))
HIFLOAT8_DTYPE = getattr(torch_npu, "hifloat8", None)


# TODO(zzzzzz198): Currently three formats(float8_e8m0fnu, float4_e2m1fn_x2, hifloat8) have to be
# specified for some operators like GMM in Ascend950, while float8_e4m3fn does not. Remove these
# filterations when operators allow to pass data with these three dtypes directly.
QUANT_DTYPES = tuple(dtype for dtype in (FLOAT4_E2M1FN_X2_DTYPE, HIFLOAT8_DTYPE) if dtype is not None)
SCALE_DTYPES = tuple(dtype for dtype in (FLOAT8_E8M0FNU_DTYPE,) if dtype is not None)


def _get_missing_symbols(symbols: tuple[str, ...]) -> list[str]:
    return [symbol for symbol in symbols if not hasattr(torch_npu, symbol)]


def _ensure_symbols_available(feature: str, symbols: tuple[str, ...]) -> None:
    missing_symbols = _get_missing_symbols(symbols)
    if not missing_symbols:
        return
    missing_symbols_str = ", ".join(missing_symbols)
    raise RuntimeError(
        f"{feature} requires a newer torch_npu runtime. Missing symbols: {missing_symbols_str}. "
        "Please upgrade torch_npu or disable MXFP quantization."
    )


def ensure_mxfp8_scale_dtype_available(feature: str) -> None:
    _ensure_symbols_available(feature, ("float8_e8m0fnu",))


def ensure_mxfp4_dtype_available(feature: str) -> None:
    _ensure_symbols_available(feature, ("float4_e2m1fn_x2", "float8_e8m0fnu"))


def ensure_mxfp8_linear_available(feature: str) -> None:
    _ensure_symbols_available(feature, ("float8_e8m0fnu", "npu_dynamic_mx_quant", "npu_quant_matmul"))


def ensure_mxfp8_moe_available(feature: str) -> None:
    _ensure_symbols_available(
        feature,
        ("float8_e8m0fnu", "npu_dynamic_mx_quant", "npu_grouped_matmul_swiglu_quant_v2"),
    )


def ensure_mxfp4_linear_available(feature: str) -> None:
    _ensure_symbols_available(
        feature, ("float4_e2m1fn_x2", "float8_e8m0fnu", "npu_dynamic_mx_quant", "npu_quant_matmul")
    )


def ensure_mxfp4_moe_available(feature: str) -> None:
    _ensure_symbols_available(
        feature,
        ("float4_e2m1fn_x2", "float8_e8m0fnu", "npu_dynamic_mx_quant", "npu_grouped_matmul_swiglu_quant_v2"),
    )


# KV cache MXFP8 scale layouts:
# K token:  [num_tokens, num_kv_heads, head_dim // 64, 2]
# K cache:  [num_blocks, num_kv_heads, block_size, head_dim // 64, 2]
# V token scale (axis=0 quant): [cdiv(num_tokens, 64), num_kv_heads, head_dim, 2]
# V cache:  [num_blocks, num_kv_heads, block_size // 64, head_dim, 2]
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
        num_kv_heads,
        block_size,
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
        num_kv_heads,
        mxfp_kv_block_scale_groups(block_size),
        head_dim,
        MXFP_KV_SCALE_VALUES_PER_GROUP,
    )


def mxfp_k_scale_numel(num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int) -> int:
    return num_blocks * mxfp_k_scale_page_bytes(num_kv_heads, block_size, head_dim)


def mxfp_v_scale_numel(num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int) -> int:
    return num_blocks * mxfp_v_scale_page_bytes(num_kv_heads, block_size, head_dim)


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


def mxfp_get_scale_dtype() -> torch.dtype:
    """Dtype used for MXFP E8M0 scale cache tensors (always 1 byte per element)."""
    if FLOAT8_E8M0FNU_DTYPE is not None:
        return FLOAT8_E8M0FNU_DTYPE
    return torch.uint8


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


def scatter_mxfp_v_scale_cache(
    value_scale: torch.Tensor,
    value_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter per-batch V scales into the paged V-scale cache.

    Matches ``modeling_qwen3_moe.py`` MXFP prefill slot derivation::

        v_scale_slot = (kv_slot_mapping // (QUANT_BLOCK_SIZE * 2)).unique()

    with ``QUANT_BLOCK_SIZE=32`` (i.e. group size 64 along the token axis).

    ``value_scale`` comes from ``npu_dynamic_mx_quant(..., axis=0)`` and has shape
    ``[ceil(num_tokens, 64), num_kv_heads, head_dim, 2]``. The cache layout is
    ``[num_blocks, num_kv_heads, block_size // 64, head_dim, 2]``.
    """
    validate_mxfp_v_scale_block_size(block_size)
    slots = slot_mapping.to(torch.long)
    num_tokens = slots.numel()
    if num_tokens == 0:
        return

    num_scale_groups = (num_tokens + MXFP_KV_SCALE_GROUP_SIZE - 1) // MXFP_KV_SCALE_GROUP_SIZE
    if value_scale.shape[0] != num_scale_groups:
        raise ValueError(
            f"C8_MXFP value_scale batch dim mismatch: got {value_scale.shape[0]}, "
            f"expected {num_scale_groups} for num_tokens={num_tokens}."
        )

    groups_per_block = mxfp_kv_block_scale_groups(block_size)
    write_group_ids = torch.arange(num_tokens, device=slots.device, dtype=torch.long)
    write_group_ids = write_group_ids // MXFP_KV_SCALE_GROUP_SIZE
    slot_groups = slots // MXFP_KV_SCALE_GROUP_SIZE

    sort_idx = torch.argsort(slot_groups, stable=True)
    sorted_groups = slot_groups[sort_idx]
    sorted_write_groups = write_group_ids[sort_idx]
    unique_mask = torch.cat(
        (
            torch.tensor([True], device=slots.device),
            sorted_groups[1:] != sorted_groups[:-1],
        )
    )
    unique_slot_groups = sorted_groups[unique_mask]
    unique_write_groups = sorted_write_groups[unique_mask]

    block_ids = unique_slot_groups // groups_per_block
    cache_group_ids = unique_slot_groups % groups_per_block
    value_scale_cache[block_ids, :, cache_group_ids, :, :] = value_scale[unique_write_groups]


# Backward-compatible aliases.
mxfp_kv_scale_cache_shape = mxfp_k_scale_cache_shape
mxfp_kv_scale_numel = mxfp_k_scale_numel
