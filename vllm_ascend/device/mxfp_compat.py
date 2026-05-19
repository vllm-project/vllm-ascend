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
# E8M0 scale elements are always 1 byte in KV cache budgeting.
MXFP_SCALE_DTYPE_SIZE = 1


def mxfp_kv_scale_groups(head_dim: int) -> int:
    if head_dim % MXFP_KV_SCALE_GROUP_SIZE != 0:
        raise ValueError(
            f"C8_MXFP KV scale cache requires head_dim divisible by {MXFP_KV_SCALE_GROUP_SIZE}, got {head_dim}."
        )
    return head_dim // MXFP_KV_SCALE_GROUP_SIZE


def mxfp_kv_block_scale_groups(block_size: int) -> int:
    if block_size % MXFP_KV_SCALE_GROUP_SIZE != 0:
        raise ValueError(
            f"C8_MXFP V scale cache requires block_size divisible by {MXFP_KV_SCALE_GROUP_SIZE}, got {block_size}."
        )
    return block_size // MXFP_KV_SCALE_GROUP_SIZE


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
    return (
        num_blocks
        * num_kv_heads
        * block_size
        * mxfp_kv_scale_groups(head_dim)
        * MXFP_KV_SCALE_VALUES_PER_GROUP
    )


def mxfp_v_scale_numel(num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int) -> int:
    return (
        num_blocks
        * num_kv_heads
        * mxfp_kv_block_scale_groups(block_size)
        * head_dim
        * MXFP_KV_SCALE_VALUES_PER_GROUP
    )


def mxfp_kv_page_size_bytes(
    block_size: int,
    num_kv_heads: int,
    k_dim: int,
    v_dim: int,
    kv_dtype_size: int,
) -> int:
    """Bytes per KV cache page for C8_MXFP (FP8 K/V tensors + E8M0 scale caches)."""
    if k_dim % MXFP_KV_SCALE_GROUP_SIZE != 0 or v_dim % MXFP_KV_SCALE_GROUP_SIZE != 0:
        raise ValueError(
            f"C8_MXFP KV cache requires K/V head dims divisible by {MXFP_KV_SCALE_GROUP_SIZE}, "
            f"got {k_dim}/{v_dim}."
        )
    mxfp_kv_block_scale_groups(block_size)
    kv_bytes = block_size * num_kv_heads * (k_dim + v_dim) * kv_dtype_size
    scale_bytes = (
        mxfp_k_scale_numel(1, block_size, num_kv_heads, k_dim)
        + mxfp_v_scale_numel(1, block_size, num_kv_heads, v_dim)
    ) * MXFP_SCALE_DTYPE_SIZE
    return kv_bytes + scale_bytes


def mxfp_get_scale_dtype() -> torch.dtype:
    """Dtype used for MXFP E8M0 scale cache tensors (always 1 byte per element)."""
    if FLOAT8_E8M0FNU_DTYPE is not None:
        return FLOAT8_E8M0FNU_DTYPE
    return torch.uint8


def mxfp_infer_head_dim_from_k_scale_numel(
    raw_k_scale_numel: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
) -> int:
    """Infer K head dim from the k_scale raw buffer and resolved num_blocks."""
    denom = (
        num_blocks
        * num_kv_heads
        * block_size
        * MXFP_KV_SCALE_VALUES_PER_GROUP
    )
    if raw_k_scale_numel % denom != 0:
        raise ValueError(
            f"C8_MXFP cannot infer k_dim from k_scale numel={raw_k_scale_numel}, "
            f"num_blocks={num_blocks}, block_size={block_size}, num_kv_heads={num_kv_heads}."
        )
    return (raw_k_scale_numel // denom) * MXFP_KV_SCALE_GROUP_SIZE


def mxfp_infer_head_dim_from_v_scale_numel(
    raw_v_scale_numel: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
) -> int:
    """Infer V head dim from the v_scale raw buffer and resolved num_blocks."""
    block_groups = mxfp_kv_block_scale_groups(block_size)
    denom = num_blocks * num_kv_heads * block_groups * MXFP_KV_SCALE_VALUES_PER_GROUP
    if raw_v_scale_numel % denom != 0:
        raise ValueError(
            f"C8_MXFP cannot infer v_dim from v_scale numel={raw_v_scale_numel}, "
            f"num_blocks={num_blocks}, block_size={block_size}, num_kv_heads={num_kv_heads}."
        )
    return raw_v_scale_numel // denom


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
    int,
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
]:
    """Derive C8_MXFP KV cache shapes from allocated raw buffer sizes only.

    Never uses page_size_bytes or KVCacheSpec head_size when they disagree with
    the split int8 buffers. Reconciles k_dim/v_dim against scale buffers when needed.

    Returns (num_blocks, k_shape, v_shape, k_scale_shape, v_scale_shape).
    """
    mxfp_kv_block_scale_groups(block_size)
    kv_slot_per_block = block_size * num_kv_heads

    def _try_match_k_dim(trial_k_dim: int) -> tuple[int, int] | None:
        if trial_k_dim % MXFP_KV_SCALE_GROUP_SIZE != 0:
            return None
        k_scale_per_block = mxfp_k_scale_numel(1, block_size, num_kv_heads, trial_k_dim)
        if raw_k_scale_numel % k_scale_per_block != 0:
            return None
        num_blocks = raw_k_scale_numel // k_scale_per_block
        k_per_block = kv_slot_per_block * trial_k_dim
        if num_blocks <= 0 or raw_k_numel != num_blocks * k_per_block:
            return None
        return num_blocks, trial_k_dim

    matching: list[tuple[int, int]] = []
    for trial_k_dim in (k_dim, v_dim):
        matched = _try_match_k_dim(trial_k_dim)
        if matched is not None:
            matching.append(matched)
    if not matching:
        max_k_dim = raw_k_numel // kv_slot_per_block if kv_slot_per_block else 0
        for trial_k_dim in range(MXFP_KV_SCALE_GROUP_SIZE, max_k_dim + 1, MXFP_KV_SCALE_GROUP_SIZE):
            matched = _try_match_k_dim(trial_k_dim)
            if matched is not None:
                matching.append(matched)
    if not matching:
        raise ValueError(
            f"C8_MXFP cannot resolve k/k_scale layout for layer={layer_name}: "
            f"raw_k_numel={raw_k_numel}, raw_k_scale_numel={raw_k_scale_numel}, "
            f"k_dim={k_dim}, v_dim={v_dim}, block_size={block_size}, num_kv_heads={num_kv_heads}."
        )

    if num_blocks_hint is not None:
        hinted = [m for m in matching if m[0] == num_blocks_hint]
        if len(hinted) == 1:
            num_blocks, resolved_k_dim = hinted[0]
        elif len(hinted) > 1:
            num_blocks, resolved_k_dim = max(hinted, key=lambda item: item[1])
        else:
            num_blocks, resolved_k_dim = min(matching, key=lambda item: item[0])
    elif len(matching) == 1:
        num_blocks, resolved_k_dim = matching[0]
    else:
        num_blocks, resolved_k_dim = min(matching, key=lambda item: item[0])

    resolved_v_dim = v_dim
    if raw_v_numel != num_blocks * kv_slot_per_block * resolved_v_dim:
        inferred_v_dim = mxfp_infer_head_dim_from_v_scale_numel(
            raw_v_scale_numel, num_blocks, block_size, num_kv_heads
        )
        if raw_v_numel != num_blocks * kv_slot_per_block * inferred_v_dim:
            raise ValueError(
                f"C8_MXFP v/v_scale layout mismatch for layer={layer_name}: "
                f"raw_v_numel={raw_v_numel}, raw_v_scale_numel={raw_v_scale_numel}, "
                f"num_blocks={num_blocks}, spec_v_dim={v_dim}, inferred_v_dim={inferred_v_dim}, "
                f"block_size={block_size}, num_kv_heads={num_kv_heads}."
            )
        resolved_v_dim = inferred_v_dim

    k_shape = (num_blocks, block_size, num_kv_heads, resolved_k_dim)
    v_shape = (num_blocks, block_size, num_kv_heads, resolved_v_dim)
    k_scale_shape = mxfp_k_scale_cache_shape(num_blocks, block_size, num_kv_heads, resolved_k_dim)
    v_scale_shape = mxfp_v_scale_cache_shape(num_blocks, block_size, num_kv_heads, resolved_v_dim)
    return num_blocks, k_shape, v_shape, k_scale_shape, v_scale_shape


def _mxfp_view_scale_cache_raw(raw_tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    expected_numel = 1
    for dim in shape:
        expected_numel *= dim
    if raw_tensor.numel() != expected_numel:
        raise ValueError(
            f"C8_MXFP scale view size mismatch: raw_numel={raw_tensor.numel()}, "
            f"shape={shape}, expected_numel={expected_numel}."
        )
    scale_dtype = mxfp_get_scale_dtype()
    return raw_tensor.view(torch.uint8).view(shape).view(scale_dtype)


def mxfp_view_k_scale_cache(
    raw_tensor: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    layer_name: str = "",
) -> torch.Tensor:
    """View k_scale raw buffer; shape is derived from numel, never passed in."""
    per_block = mxfp_k_scale_numel(1, block_size, num_kv_heads, head_dim)
    if raw_tensor.numel() % per_block != 0:
        raise ValueError(
            f"C8_MXFP k_scale buffer size mismatch for layer={layer_name}: "
            f"numel={raw_tensor.numel()}, per_block={per_block}, block_size={block_size}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}."
        )
    num_blocks = raw_tensor.numel() // per_block
    shape = mxfp_k_scale_cache_shape(num_blocks, block_size, num_kv_heads, head_dim)
    return _mxfp_view_scale_cache_raw(raw_tensor, shape)


def mxfp_view_v_scale_cache(
    raw_tensor: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    layer_name: str = "",
) -> torch.Tensor:
    """View v_scale raw buffer; shape is derived from numel, never passed in."""
    per_block = mxfp_v_scale_numel(1, block_size, num_kv_heads, head_dim)
    if raw_tensor.numel() % per_block != 0:
        raise ValueError(
            f"C8_MXFP v_scale buffer size mismatch for layer={layer_name}: "
            f"numel={raw_tensor.numel()}, per_block={per_block}, block_size={block_size}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}."
        )
    num_blocks = raw_tensor.numel() // per_block
    shape = mxfp_v_scale_cache_shape(num_blocks, block_size, num_kv_heads, head_dim)
    return _mxfp_view_scale_cache_raw(raw_tensor, shape)


def mxfp_view_scale_cache(raw_tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Backward-compatible wrapper; prefer mxfp_view_k/v_scale_cache."""
    return _mxfp_view_scale_cache_raw(raw_tensor, shape)


# Backward-compatible aliases.
mxfp_kv_scale_cache_shape = mxfp_k_scale_cache_shape
mxfp_kv_scale_numel = mxfp_k_scale_numel
