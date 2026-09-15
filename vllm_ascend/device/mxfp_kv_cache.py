import torch
import torch_npu

# KV cache MXFP8 scale layouts, PA_NZ flavor (QFA layout_kv="PA_NZ"; golden
# reference: quant_flash_attn_golden.py "PA_NZ: fp8=[Bn,N,D//32,Bs,32],
# Kscale=[Bn,N,Bs//16,D//64,16,2], Vscale=[Bn,N,D//16,Bs//64,16,2]"). The K/V
# caches themselves keep the natural [num_blocks, block_size, num_kv_heads,
# head_dim] storage and are viewed as (num_blocks, num_kv_heads,
# head_dim // 32, block_size, 32) at the operator boundary -- the same NZ
# view npu_scatter_pa_kv_cache already consumes on the FIA C8 path -- so
# allocation, hybrid partitioning, PD transfer and prefix-cache CoW are all
# layout-agnostic. Only the scale caches below change physical shape.
# The trailing 2 packs the even/odd 32-element MX scale groups (the native
# output format of npu_dynamic_mx_quant; unchanged by PA_NZ, which only
# rearranges the outer axes).
# K scale token:  [num_tokens, num_kv_heads, head_dim // 64, 2]
# K scale cache:  [num_blocks, num_kv_heads, block_size // 16, head_dim // 64, 16, 2]
# V scale token (axis=0 quant): [cdiv(num_tokens, 64), num_kv_heads, head_dim, 2]
# V scale cache:  [num_blocks, num_kv_heads, head_dim // 16, block_size // 64, 16, 2]
MXFP_KV_SCALE_GROUP_SIZE = 64
MXFP_KV_SCALE_VALUES_PER_GROUP = 2
MXFP_KV_NZ_DIM_FRAG = 32
MXFP_K_SCALE_NZ_TOKEN_FRAG = 16
MXFP_V_SCALE_NZ_DIM_FRAG = 16
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
) -> tuple[int, int, int, int, int, int]:
    return (
        num_blocks,
        num_kv_heads,
        block_size // MXFP_K_SCALE_NZ_TOKEN_FRAG,
        mxfp_kv_scale_groups(head_dim),
        MXFP_K_SCALE_NZ_TOKEN_FRAG,
        MXFP_KV_SCALE_VALUES_PER_GROUP,
    )


def mxfp_v_scale_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int, int, int, int]:
    return (
        num_blocks,
        num_kv_heads,
        head_dim // MXFP_V_SCALE_NZ_DIM_FRAG,
        mxfp_kv_block_scale_groups(block_size),
        MXFP_V_SCALE_NZ_DIM_FRAG,
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
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int, int, int],
]:
    """Derive C8_MXFP KV cache shapes from spec dims and allocated raw buffer sizes.

    ``num_blocks`` is derived from the k_scale buffer; ``k_dim``/``v_dim`` come from the caller
    (typically ``KVCacheSpec``). All four raw buffers must match the expected numel.

    The K/V caches keep the natural 4-D (num_blocks, block_size, num_kv_heads,
    dim) storage; the NZ 5-D view for QFA's PA_NZ layout is taken at the
    operator boundary. The scale caches are allocated directly in the 6-D
    PA_NZ shapes.

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


def scatter_mxfp_pa_nz_kv_cache(
    quant_key: torch.Tensor,
    quant_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter quantized K/V into the paged caches in QFA's PA_NZ layout.

    The caches keep the natural (num_blocks, block_size, num_kv_heads,
    head_dim) storage; both this scatter and QFA (layout_kv=PA_NZ) go through
    the 5-D view (num_blocks, num_kv_heads, head_dim//32, block_size, 32).
    A token at in-block offset ``o`` lands at ``[block, :, :, o, :]`` and its
    ``[N, D]`` payload needs only a reshape to ``[N, D//32, 32]`` -- the NZ
    layout is exactly the natural token row re-fragmented along D, so no
    permute is involved. npu_scatter_pa_kv_cache cannot be used here: its
    shape contract requires key_cache.dim2 == num_kv_heads, which the PA_NZ
    axis order (dim2 == head_dim//32) violates.

    Byte views throughout: index_put on float8 either errors or falls back
    to AICPU. Padded rows (slot -1) are clamped to slot 0 and rewrite the
    cache's pre-read content, making them no-ops (same pattern as the K-scale
    scatter).
    """
    slots = slot_mapping.to(torch.long)
    if slots.numel() == 0:
        return

    valid = slots >= 0
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    block_ids = safe_slots // block_size
    offsets = safe_slots % block_size

    for src, cache in ((quant_key, key_cache), (quant_value, value_cache)):
        num_tokens, num_kv_heads, head_dim = src.shape
        src_bytes = src.view(torch.uint8) if src.dtype != torch.uint8 else src
        # aclnnIndex/aclnnIndexPut reject DT_FLOAT8_E4M3FN outright, so the
        # cache side needs the byte view just like the payload side.
        cache_bytes = cache.view(torch.uint8) if cache.dtype != torch.uint8 else cache
        nz = cache_bytes.view(
            -1,
            num_kv_heads,
            head_dim // MXFP_KV_NZ_DIM_FRAG,
            block_size,
            MXFP_KV_NZ_DIM_FRAG,
        )
        payload = src_bytes.view(
            num_tokens,
            num_kv_heads,
            head_dim // MXFP_KV_NZ_DIM_FRAG,
            MXFP_KV_NZ_DIM_FRAG,
        )
        cached = nz[block_ids, :, :, offsets, :]
        updates = torch.where(valid.view(-1, 1, 1, 1), payload, cached)
        nz[block_ids, :, :, offsets, :] = updates


def scatter_mxfp_k_scale_cache(
    key_scale: torch.Tensor,
    key_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter per-token K scales into the paged K-scale cache.

    ``key_scale`` shape: ``[num_tokens, num_kv_heads, head_dim // 64, 2]``
    (any 1-byte dtype; callers pass a uint8 view of the E8M0 scale).
    ``key_scale_cache`` shape (PA_NZ): ``[num_blocks, num_kv_heads,
    block_size // 16, head_dim // 64, 16, 2]`` -- a token at in-block offset
    ``o`` lands at ``[block, n, o // 16, :, o % 16, :]``.

    ACL-graph-capture safe: no host-device synchronization (``.all()``/
    ``bool()``/``.item()`` are illegal mid-capture -- "Stream during the
    capture stage is not supported") and no data-dependent shapes. Padded
    rows (slot -1) are clamped to slot 0 via ``torch.where`` and write back
    the cache's pre-read content, making them no-ops. Known edge (unreachable
    in supported paths): a real token targeting slot 0 IN THE SAME BATCH as a
    padded row would be a duplicate-index write where the padding row's
    read-back clobbers the real value -- eager batches never carry -1 rows,
    and graph-mode padding uses valid dummy slots, so this combination cannot
    occur in v1.
    """
    validate_mxfp_v_scale_block_size(block_size)
    slots = slot_mapping.to(torch.long)
    if slots.numel() == 0:
        return

    valid = slots >= 0
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    block_ids = safe_slots // block_size
    offsets = safe_slots % block_size
    seg_ids = offsets // MXFP_K_SCALE_NZ_TOKEN_FRAG
    frag_ids = offsets % MXFP_K_SCALE_NZ_TOKEN_FRAG
    # Row mask (device-only): valid rows take the new scale, padded rows
    # rewrite the current content of their clamp target -- a no-op.
    cached = key_scale_cache[block_ids, :, seg_ids, :, frag_ids, :]
    updates = torch.where(valid.view(-1, 1, 1, 1), key_scale, cached)
    key_scale_cache[block_ids, :, seg_ids, :, frag_ids, :] = updates


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
    PA_NZ ``[num_blocks, num_kv_heads, head_dim // 16, block_size // 64, 16, 2]``:
    a 64-token group at in-block group-offset ``g`` lands at
    ``[block, n, d // 16, g, d % 16, :]`` for every channel ``d``.

    Unused while V's scale is the checkpoint's static per-channel one (a
    static V scale is broadcast into the cache once and never scattered);
    kept for a dynamic-V design. Indexing follows the PA_NZ order the rest
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
    # value_scale: [G, N, D, 2] -> [G, N, D // 16, 16, 2]; the advanced
    # indexing below broadcasts the G-length index vectors to the front,
    # so the source lines up group-by-group with the target slots.
    packed = value_scale.reshape(num_scales, *value_scale.shape[1:2], -1, MXFP_V_SCALE_NZ_DIM_FRAG, MXFP_KV_SCALE_VALUES_PER_GROUP)
    value_scale_cache[block_ids, :, :, v_scale_cache_offsets, :, :] = packed
