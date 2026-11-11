# SPDX-License-Identifier: Apache-2.0
"""DSpark multi-layer context KV normalization, interleaved RoPE and SWA store."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _store_layer(
    kv, weight, cos, sin, cache, slots, token,
    KV_TOKEN: tl.constexpr,
    COS_TOKEN: tl.constexpr, SIN_TOKEN: tl.constexpr,
    CACHE_BLOCK: tl.constexpr, CACHE_TOKEN: tl.constexpr,
    SLOT_TOKEN: tl.constexpr, SLOT_COORD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr, NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr, SLOT_PAIRS: tl.constexpr,
    EPS: tl.constexpr,
):
    dim = tl.arange(0, HEAD_DIM)
    x = tl.load(kv + token * KV_TOKEN + dim).to(tl.float32)
    w = tl.load(weight + dim).to(tl.float32)
    norm = tl.rsqrt(tl.sum(x * x, 0) / HEAD_DIM + EPS)
    # The original RMSNorm materializes a BF16 row before rotary_mul.
    value = (x * norm * w).to(kv.dtype.element_ty).to(tl.float32)
    mate = tl.where((dim - NOPE_DIM) % 2 == 0, dim + 1, dim - 1)
    mate = tl.minimum(tl.maximum(mate, 0), HEAD_DIM - 1)
    paired = tl.gather(value, mate, 0)
    c = tl.load(cos + token * COS_TOKEN + tl.minimum(tl.maximum(dim - NOPE_DIM, 0), ROPE_DIM - 1),
                dim >= NOPE_DIM, other=1).to(tl.float32)
    s = tl.load(sin + token * SIN_TOKEN + tl.minimum(tl.maximum(dim - NOPE_DIM, 0), ROPE_DIM - 1),
                dim >= NOPE_DIM, other=0).to(tl.float32)
    rotated = value * c + tl.where((dim - NOPE_DIM) % 2 == 0, -paired, paired) * s
    output = tl.where(dim >= NOPE_DIM, rotated, value).to(kv.dtype.element_ty)
    if SLOT_PAIRS:
        block = tl.load(slots + token * SLOT_TOKEN).to(tl.int64)
        offset = tl.load(slots + token * SLOT_TOKEN + SLOT_COORD).to(tl.int64)
    else:
        slot = tl.load(slots + token * SLOT_TOKEN).to(tl.int64)
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
    valid = (block >= 0) & (block < NUM_BLOCKS) & (offset >= 0) & (offset < BLOCK_SIZE)
    if valid:
        tl.store(cache + block * CACHE_BLOCK + offset * CACHE_TOKEN + dim, output)


@triton.jit(do_not_specialize=["num_tokens"])
def _dspark_context_kv_kernel(
    kv, weights, cosines, sines, caches, slots, num_tokens,
    KV_TOKEN: tl.constexpr, KV_LAYER: tl.constexpr,
    COS_TOKENS: tl.constexpr, SIN_TOKENS: tl.constexpr,
    CACHE_BLOCKS: tl.constexpr, CACHE_TOKENS: tl.constexpr,
    SLOT_TOKENS: tl.constexpr, SLOT_COORDS: tl.constexpr,
    BLOCK_SIZES: tl.constexpr, NUM_BLOCKS: tl.constexpr,
    SLOT_PAIRS: tl.constexpr, EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, NOPE_DIM: tl.constexpr, ROPE_DIM: tl.constexpr,
    NUM_LAYERS: tl.constexpr, LAYER_INDICES: tl.constexpr,
):
    layer = tl.program_id(1)
    for token in range(tl.program_id(0), num_tokens, tl.num_programs(0)):
        for i in tl.static_range(NUM_LAYERS):
            if layer == i:
                _store_layer(
                    kv + LAYER_INDICES[i] * KV_LAYER, weights[i], cosines[i], sines[i], caches[i], slots[i],
                    token, KV_TOKEN, COS_TOKENS[i], SIN_TOKENS[i],
                    CACHE_BLOCKS[i], CACHE_TOKENS[i], SLOT_TOKENS[i], SLOT_COORDS[i],
                    BLOCK_SIZES[i], NUM_BLOCKS[i], HEAD_DIM, NOPE_DIM, ROPE_DIM,
                    SLOT_PAIRS[i], EPS[i],
                )


def can_fuse_dspark_context_kv(kv, layers, slot_mappings, rope) -> bool:
    """Only accept the standard BF16, single-head, paged A2/A3 layout."""
    if kv.ndim != 3 or kv.device.type != "npu" or kv.dtype != torch.bfloat16 or kv.stride(2) != 1:
        return False
    if not 2 <= len(layers) <= 4 or len(slot_mappings) != len(layers):
        return False
    nope_dim = layers[0].self_attn.nope_head_dim
    rope_dim = layers[0].self_attn.rope_head_dim
    for layer, slots in zip(layers, slot_mappings, strict=True):
        attn = layer.self_attn
        cache_layer = attn.dsa_attn.swa_cache_layer
        cache = cache_layer.kv_cache[0]
        if (
            slots is None or attn.packed_cache_ops is not None
            or slots.ndim not in (1, 2) or slots.shape[0] < kv.shape[0]
            or (slots.ndim == 2 and slots.shape[1] != 2)
            or slots.dtype not in (torch.int32, torch.int64)
            or slots.device != kv.device or cache.device != kv.device
            or slots.stride(-1) != 1
            or cache.ndim != 4 or cache.shape[2] != 1
            or cache.shape[3] != kv.shape[2] or cache.dtype != kv.dtype
            or cache.stride(3) != 1 or cache_layer.block_size != cache.shape[1]
            or attn.kv_norm.weight.ndim != 1
            or attn.kv_norm.weight.shape[0] != kv.shape[2]
            or attn.kv_norm.weight.stride(0) != 1
            or attn.kv_norm.weight.device != kv.device
            or attn.nope_head_dim + attn.rope_head_dim != kv.shape[2]
            or attn.rope_head_dim % 2 != 0
            or attn.nope_head_dim != nope_dim or attn.rope_head_dim != rope_dim
            or kv.shape[2] & (kv.shape[2] - 1)
        ):
            return False
        name = attn.rotary_emb.layername
        cos, sin = rope
        if (
            cos[name].shape[0] < kv.shape[0] or sin[name].shape[0] < kv.shape[0]
            or cos[name].shape[-1] != attn.rope_head_dim
            or sin[name].shape[-1] != attn.rope_head_dim
            or cos[name].stride(-1) != 1 or sin[name].stride(-1) != 1
            or cos[name].device != kv.device or sin[name].device != kv.device
        ):
            return False
    return True


def fused_dspark_context_kv(kv, layers, slot_mappings, rope, layer_indices) -> None:
    """Process all draft layers in one launch without allocating KV intermediates."""
    attns = [layer.self_attn for layer in layers]
    caches = [attn.dsa_attn.swa_cache_layer.kv_cache[0] for attn in attns]
    cos, sin = rope
    cosines = [cos[attn.rotary_emb.layername] for attn in attns]
    sines = [sin[attn.rotary_emb.layername] for attn in attns]
    init_device_properties_triton()
    programs = min(kv.shape[0], max(1, get_vectorcore_num() // len(layers)))
    _dspark_context_kv_kernel[(programs, len(layers))](
        kv, tuple(attn.kv_norm.weight for attn in attns), tuple(cosines), tuple(sines),
        tuple(caches), tuple(slot_mappings), kv.shape[0],
        kv.stride(0), kv.stride(1),
        tuple(t.stride(0) for t in cosines), tuple(t.stride(0) for t in sines),
        tuple(t.stride(0) for t in caches), tuple(t.stride(1) for t in caches),
        tuple(t.stride(0) for t in slot_mappings),
        tuple(t.stride(1) if t.ndim == 2 else 0 for t in slot_mappings),
        tuple(attn.dsa_attn.swa_cache_layer.block_size for attn in attns),
        tuple(t.shape[0] for t in caches),
        tuple(t.ndim == 2 for t in slot_mappings),
        tuple(attn.kv_norm.variance_epsilon for attn in attns),
        kv.shape[2], attns[0].nope_head_dim, attns[0].rope_head_dim, len(layers), tuple(layer_indices),
    )
