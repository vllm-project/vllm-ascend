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


# Backward-compatible aliases.
mxfp_kv_scale_cache_shape = mxfp_k_scale_cache_shape
mxfp_kv_scale_numel = mxfp_k_scale_numel
