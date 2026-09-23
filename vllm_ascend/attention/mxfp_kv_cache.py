import math

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

    This is scenario 1 of the ScatterPaKvCache contract (ops-transformer
    ``attention/scatter_pa_kv_cache/README.md``), which is the operator's
    native PA_NZ mode::

        key/value   [batch * seq_len, num_head, head_size]
        key/valueCache
                    [num_blocks, num_head, head_size // last_dim, block_size, last_dim]
        slotMapping [batch * seq_len]
        cacheMode   "PA_NZ"
        last_dim = 32 / sizeof(dtype)     -> 32 for any 1-byte dtype
        (head_size * sizeof(dtype)) % 32 == 0

    ``cache_mode`` is what selects that contract and is not optional: left at
    the default the operator reads the caches as scenario 2 ("Norm",
    ``[num_blocks, block_size, num_head, head_size]``), whose dim2 is
    num_head -- which is where the "shape contract requires
    key_cache.dim2 == num_kv_heads" rejection comes from. The axis order is
    not the problem; omitting the mode is.

    Both sides go through int8 views. The operator does accept FLOAT8_E4M3FN,
    but only on Ascend 950PR/950DT -- A2/A3 are limited to FP16/BF16/INT8,
    and INT8 is accepted everywhere. sizeof is 1 either way, so last_dim
    stays 32 and the bytes written are identical; this just keeps one fewer
    product-dependent assumption in the call.

    Negative slots (vLLM's PAD_SLOT_ID) are left for the operator to skip, so
    the padded-batch case costs nothing and the shapes stay static for graph
    capture.

    Constraints checked against this model: head_dim 256 -> 256 % 32 == 0,
    and block_size (the cache's second-to-last axis under PA_NZ) must stay
    below UINT16_MAX.
    """
    if slot_mapping.numel() == 0:
        return

    num_kv_heads, head_dim = quant_key.shape[1], quant_key.shape[2]

    def _as_bytes(t: torch.Tensor) -> torch.Tensor:
        return t if t.dtype == torch.int8 else t.view(torch.int8)

    def _nz_view(cache: torch.Tensor) -> torch.Tensor:
        return _as_bytes(cache).view(
            -1,
            num_kv_heads,
            head_dim // MXFP_KV_NZ_DIM_FRAG,
            block_size,
            MXFP_KV_NZ_DIM_FRAG,
        )

    torch_npu.npu_scatter_pa_kv_cache(
        key=_as_bytes(quant_key),
        value=_as_bytes(quant_value),
        key_cache=_nz_view(key_cache),
        value_cache=_nz_view(value_cache),
        slot_mapping=slot_mapping,
        cache_mode="PA_NZ",
    )


def fill_mxfp_v_scale_cache(value_scale: torch.Tensor, value_scale_cache: torch.Tensor) -> None:
    """Broadcast V's static per-channel E8M0 scale over its whole paged cache.

    ``value_scale`` is the checkpoint's flat ``(num_kv_heads * head_dim,)``
    E8M0 byte vector, ``value_scale_cache`` the PA_NZ 6-D cache
    ``[num_blocks, num_kv_heads, head_dim // 16, block_size // 64, 16, 2]``.
    Reshaping the source to ``(num_kv_heads, head_dim // 16, 1, 16, 1)`` lines
    its channel axis up with the cache's fragment split and lets the block,
    token-group and even/odd axes broadcast.

    V's scale is static, so unlike K's this is a one-time fill rather than a
    per-step scatter -- the caller runs it once the caches exist, before any
    request, capture or replay.

    Head count and V head dim are read from the cache rather than the layer so
    a model whose V head dim differs from Q/K's stays correct.
    """
    num_kv_heads = value_scale_cache.shape[1]
    v_dim_frags = value_scale_cache.shape[2]
    v_dim_frag_size = value_scale_cache.shape[4]
    value_scale_cache.copy_(value_scale.view(num_kv_heads, v_dim_frags, 1, v_dim_frag_size, 1))


def mxfp_k_scale_slot_index(
    slot_mapping: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose slots into the index tensors the K-scale scatter indexes with.

    Returns ``(block_ids, seg_ids, frag_ids)`` for the PA_NZ K-scale cache,
    where a token at in-block offset ``o`` lands at
    ``[block, n, o // 16, :, o % 16, :]``.

    Depends only on per-step data (slot_mapping, block_size), so one call
    serves every full-attention layer of a step -- see
    ``AscendC8MXFPAttentionBackendImpl._qfa_k_scale_slot_index`` for the
    caching, including why caching this one through graph capture is safe
    while the metadata-op plan is not.

    Padded rows (slot -1) are clamped to slot 0, which keeps the shapes
    static: no ``.item()``, no boolean indexing, nothing a capture would
    reject. Slot 0 is theirs to take -- see scatter_mxfp_k_scale_cache.
    """
    safe_slots = slot_mapping.to(torch.long).clamp(min=0)
    block_ids = safe_slots // block_size
    offsets = safe_slots % block_size
    return block_ids, offsets // MXFP_K_SCALE_NZ_TOKEN_FRAG, offsets % MXFP_K_SCALE_NZ_TOKEN_FRAG


def scatter_mxfp_k_scale_cache(
    key_scale: torch.Tensor,
    key_scale_cache: torch.Tensor,
    slot_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Scatter per-token K scales into the paged K-scale cache.

    ``key_scale`` shape: ``[num_tokens, num_kv_heads, head_dim // 64, 2]``
    (any 1-byte dtype; callers pass a uint8 view of the E8M0 scale).
    ``key_scale_cache`` shape (PA_NZ): ``[num_blocks, num_kv_heads,
    block_size // 16, head_dim // 64, 16, 2]``.

    ``slot_index`` comes from :func:`mxfp_k_scale_slot_index` and is shared
    across the step's layers; only the write below is per-layer.

    ACL-graph-capture safe: no host-device synchronization (``.all()``/
    ``bool()``/``.item()`` are illegal mid-capture -- "Stream during the
    capture stage is not supported") and no data-dependent shapes.

    Padded rows arrive clamped to slot 0 and are simply written there. Slot 0
    belongs to block 0, which vLLM's BlockPool takes out of the free list as
    the null block and never hands to a request, so no real token's scale can
    live there; what lands in it is read back only by the dummy requests that
    pad a captured batch, whose output is discarded. This used to gather the
    cache, select the old content for padded rows and write the result back --
    three kernels in every full-attention layer of every step to make rows
    that only a FULL-graph replay, an MTP verify or a draft step ever carries
    into a no-op. Eager batches are sliced to their real token count and
    never had such rows, so for them the gather and the select were an
    identity on top of the write.
    """
    block_ids, seg_ids, frag_ids = slot_index
    if block_ids.numel() == 0:
        return
    key_scale_cache[block_ids, :, seg_ids, :, frag_ids, :] = key_scale


def split_hybrid_c8_mxfp_cache_buffer(
    raw_tensor: torch.Tensor,
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a hybrid Mamba/attention buffer into C8 MXFP payloads.

    Hybrid cache groups allocate one padded raw buffer that is shared by
    Mamba and full-attention layers. Mamba state views start at the front
    of that buffer, while attention payloads are placed at the end. C8
    MXFP needs four payloads instead of the regular K/V pair.
    """
    num_blocks, block_size, num_kv_heads, k_dim = k_shape
    if len(v_shape) != 4:
        raise ValueError(f"Expected a four-dimensional V cache shape, got {v_shape}.")
    if v_shape[:3] != k_shape[:3]:
        raise ValueError(
            "C8_MXFP hybrid K/V cache shapes must share block and head dimensions, "
            f"got k_shape={k_shape}, v_shape={v_shape}."
        )

    v_dim = v_shape[3]
    k_scale_shape = mxfp_k_scale_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        k_dim,
    )
    v_scale_shape = mxfp_v_scale_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        v_dim,
    )
    payload_sizes = [
        math.prod(k_shape),
        math.prod(v_shape),
        math.prod(k_scale_shape),
        math.prod(v_scale_shape),
    ]
    payload_size = sum(payload_sizes)
    if raw_tensor.numel() < payload_size:
        raise ValueError(
            "C8_MXFP hybrid cache buffer is too small: "
            f"raw_numel={raw_tensor.numel()}, payload_numel={payload_size}, "
            f"k_shape={k_shape}, v_shape={v_shape}."
        )

    payload = raw_tensor[raw_tensor.numel() - payload_size :]
    raw_k, raw_v, raw_k_scale, raw_v_scale = torch.split(payload, payload_sizes)
    return raw_k, raw_v, raw_k_scale, raw_v_scale
