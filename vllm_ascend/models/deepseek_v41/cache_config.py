# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 layer-outermost cache placement and allocation."""

import math
from dataclasses import dataclass, replace

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.device.hardware_profile import builds_scatter_nd_update_sk

STATE_RING_ROWS = 32
MAX_VALIDATED_SLOT_BYTES = 10 * 1024**3


@dataclass(frozen=True, kw_only=True)
class DeepseekV41FullSpec(AscendMLAAttentionSpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and s.block_size == self.block_size
            and getattr(s, "tokens_per_state", 1) in (1, 2)
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41IndexerSpec(AscendMLAAttentionSpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and s.block_size == self.block_size
            and getattr(s, "tokens_per_state", 1) in (1, 2)
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41SWASpec(AscendSlidingWindowMLASpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41SWASpec) and s.sliding_window == self.sliding_window for s in specs.values()
        )


def is_v41_spec(spec):
    return isinstance(spec, (DeepseekV41FullSpec, DeepseekV41IndexerSpec, DeepseekV41SWASpec)) or (
        getattr(spec, "model_version", None) == "deepseek_v41"
    )


def is_deepseek_v41_cache(specs_or_groups):
    if isinstance(specs_or_groups, dict):
        specs = list(specs_or_groups.values())
    else:
        specs = []
        for item in specs_or_groups:
            spec = getattr(item, "kv_cache_spec", item)
            if isinstance(spec, UniformTypeKVCacheSpecs):
                specs.extend(spec.kv_cache_specs.values())
            else:
                specs.append(spec)
    return any(is_v41_spec(spec) for spec in specs)


def _layer_number(name):
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def _draft_layer_number(name):
    return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])


def _cache_plane_sizes(spec):
    """Payload bytes per page for each plane, matching DSV4F plan_cache_slots."""
    from vllm_ascend.core.kv_cache_interface import get_storage_block_size

    num_kv_heads = getattr(spec, "num_kv_heads", None)
    head_size = getattr(spec, "head_size", None)
    if num_kv_heads is None or head_size is None:
        return (int(spec.unpadded_page_size_bytes),)
    storage = int(getattr(spec, "storage_block_size", None) or get_storage_block_size(spec))
    rows = storage * int(num_kv_heads)
    key_bytes = rows * int(head_size) * spec.dtype.itemsize
    scale_dim = int(getattr(spec, "scale_dim", 0) or 0)
    if scale_dim:
        return key_bytes, rows * scale_dim * spec.scale_dtype.itemsize
    return (key_bytes,)


def _cache_plane_row_bytes(spec):
    """Return the byte width of one token row for each plane the spec stores."""
    num_kv_heads = getattr(spec, "num_kv_heads", None)
    head_size = getattr(spec, "head_size", None)
    if num_kv_heads is None or head_size is None:
        return (spec.unpadded_page_size_bytes,)
    key_bytes = num_kv_heads * head_size * spec.dtype.itemsize
    scale_dim = getattr(spec, "scale_dim", 0)
    if scale_dim:
        return key_bytes, num_kv_heads * scale_dim * spec.scale_dtype.itemsize
    return (key_bytes,)


def _flat_row_page_size(capacity, slot_specs):
    """Round a slot page up until every plane's rows tile it without drift.

    Where ``scatter_nd_update_sk`` is unavailable the store addresses a plane
    through a contiguous ``[rows, width]`` view of the slot backing. That view
    only coincides with the strided per-page view when consecutive pages start
    on a row boundary for every plane sharing the slot, which needs the page to
    be a common multiple of their row sizes.
    """
    alignment = 1
    for spec in slot_specs:
        for row_bytes in _cache_plane_row_bytes(spec):
            alignment = math.lcm(alignment, row_bytes)
    return -(-capacity // alignment) * alignment


def get_layer_tuples(specs):
    """Return DSV4-style ordered layer tuples and their physical page sizes."""
    mla = {name for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec)}
    state = sorted((name for name, spec in specs.items() if isinstance(spec, CircularBufferSpec)), key=_layer_number)
    swa = {name for name, spec in specs.items() if isinstance(spec, AscendSlidingWindowMLASpec)}

    full = sorted((name for name in mla if not specs[name].scale_dim), key=_layer_number)
    target_swa = sorted((name for name in swa if ".mtp." not in f".{name}"), key=_layer_number)
    draft_swa = sorted((name for name in swa if ".mtp." in f".{name}"), key=_draft_layer_number)

    layer_tuples: list[tuple[str, ...]] = []
    page_sizes: list[int] = []
    for slot_idx, kv_name in enumerate(full):
        prefix = kv_name.rsplit(".", 1)[0]
        index_name = prefix + ".indexer.k_cache"
        index_spec = specs[index_name]
        kv_spec = specs[kv_name]
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + target_swa[slot_idx :: len(full)]
        kv_bytes = sum(_cache_plane_sizes(kv_spec))
        index_bytes = sum(_cache_plane_sizes(index_spec))
        alias_bytes = max((sum(_cache_plane_sizes(specs[name])) for name in aliases), default=0)
        # Attention addresses an SWA page as one dense payload, so on the SoCs
        # that take the dense-store path the index plane may only stay in the
        # source page while it fits the slack the aliases already pay for.
        # Otherwise it gets a page of its own: sharing makes SparseFlashMla
        # read index bytes as KV, and generation then runs to the token cap
        # producing nothing. Elsewhere the page is shared, as upstream lays it
        # out (DSV4F plan_cache_slots).
        share_index = builds_scatter_nd_update_sk() or kv_bytes + index_bytes <= alias_bytes
        capacity = max(kv_bytes + index_bytes, alias_bytes) if share_index else max(kv_bytes, alias_bytes)
        if slot_idx < len(draft_swa):
            aliases.append(draft_swa[slot_idx])
        if not builds_scatter_nd_update_sk():
            align_specs = [kv_spec, *(specs[name] for name in aliases)] + ([index_spec] if share_index else [])
            aligned = _flat_row_page_size(capacity, align_specs)
            if aligned != capacity and any(isinstance(specs[name], CircularBufferSpec) for name in aliases):
                raise ValueError(
                    f"V4.1 slot {slot_idx} carries the compressor ring, which must fill its page exactly, "
                    f"so its {capacity}-byte page cannot be aligned up to {aligned} for the dense store"
                )
            capacity = aligned
        if share_index:
            layer_tuples.append((kv_name, index_name, *aliases))
            page_sizes.append(capacity)
        else:
            layer_tuples.append((kv_name, *aliases))
            page_sizes.append(capacity)
            index_page = index_bytes
            if not builds_scatter_nd_update_sk():
                index_page = _flat_row_page_size(index_bytes, [index_spec])
            layer_tuples.append((index_name,))
            page_sizes.append(index_page)
    return page_sizes, layer_tuples


def group_cache_specs(specs):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    page_sizes, layer_tuples = get_layer_tuples(specs)
    # DSV4F plan_cache_slots: KV advertises kv_bytes, shared index the slack,
    # aliases the full physical page.
    padded = {}
    for page_size, layer_tuple in zip(page_sizes, layer_tuples):
        if len(layer_tuple) == 1 and getattr(specs[layer_tuple[0]], "scale_dim", None):
            padded[layer_tuple[0]] = replace(specs[layer_tuple[0]], page_size_padded=page_size)
            continue
        kv_name, *rest = layer_tuple
        kv_bytes = sum(_cache_plane_sizes(specs[kv_name]))
        padded[kv_name] = replace(specs[kv_name], page_size_padded=kv_bytes)
        if rest and getattr(specs[rest[0]], "scale_dim", 0):
            index_name, *rest = rest
            padded[index_name] = replace(specs[index_name], page_size_padded=page_size - kv_bytes)
        for name in rest:
            padded[name] = replace(specs[name], page_size_padded=page_size)

    mla_names = [name for name, spec in padded.items() if isinstance(spec, AscendMLAAttentionSpec)]
    state_names = [name for name, spec in padded.items() if isinstance(spec, CircularBufferSpec)]
    groups = [
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in mla_names}),
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in state_names}),
    ]

    # Transpose the physical tuples. Each scheduler SWA group takes one layer
    # from every tuple, so its members use distinct slots at the same block ID.
    swa_columns = [
        [
            name
            for name in layer_tuple
            if isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." not in f".{name}"
        ]
        for layer_tuple in layer_tuples
        if any(
            isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." not in f".{name}" for name in layer_tuple
        )
    ]
    for names in zip(*swa_columns):
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in names}))

    draft_names = [
        name
        for layer_tuple in layer_tuples
        for name in layer_tuple
        if isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." in f".{name}"
    ]
    if draft_names:
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in draft_names}))
    return groups


def make_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


def _specs_from_groups(groups):
    specs = {}
    for group in groups:
        for name in group.layer_names:
            specs[name] = group.kv_cache_spec.kv_cache_specs[name]
    return specs


def get_deepseek_v41_pool_bytes_per_block(groups):
    page_sizes, _ = get_layer_tuples(_specs_from_groups(groups))
    return sum(page_sizes)


def addressing_safe_num_blocks(page_sizes: list[int], num_blocks: int) -> int:
    """Cap the pool so no slot tensor outgrows its validated addressing range."""
    if builds_scatter_nd_update_sk():
        return num_blocks
    widest_page = max(page_sizes)
    safe_blocks = MAX_VALIDATED_SLOT_BYTES // widest_page
    if safe_blocks >= num_blocks:
        return num_blocks
    logger.warning(
        "V4.1 caps the KV cache at %d blocks instead of %d: this SoC lacks "
        "scatter_nd_update_sk, and its dense store only addresses slots up to "
        "%d bytes correctly (widest slot page %d bytes). Raising the cap "
        "silently corrupts every request until the server restarts.",
        safe_blocks,
        num_blocks,
        MAX_VALIDATED_SLOT_BYTES,
        widest_page,
    )
    return safe_blocks


def get_deepseek_v41_kv_cache_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Allocate four independent layer slots backed by one global block-ID pool."""
    page_sizes, layer_tuples = get_layer_tuples(_specs_from_groups(groups))
    capacity = max(available_memory // sum(page_sizes), 0)
    num_blocks = addressing_safe_num_blocks(page_sizes, may_override_num_blocks(vllm_config, capacity))
    tensors: list[KVCacheTensor] = []
    for page_size, layer_names in zip(page_sizes, layer_tuples):
        size = num_blocks * page_size
        tensors.append(
            KVCacheTensor(
                size=size,
                layers=list(layer_names),
                offset=0,
                layer_stride=0,
                block_stride=page_size,
            )
        )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )


def pin_v41_attn_kv_dtype(vllm_config) -> None:
    """Pin V4.1 attention KV to BF16 without a launch flag.

    Inherited DSV4 modules read cache_dtype through get_dsv4_attn_kv_dtype(),
    which treats ``auto`` as FP8 on A5. SparseFlashMla and DSpark SWA are BF16
    PA_BBND, so an FP8 plan behind those planes garbles decode and zeros later
    draft positions. DSV4F does this in validate_cache_runtime.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        return
    cache_dtype = str(getattr(cache_config, "cache_dtype", "auto")).lower()
    if cache_dtype not in ("auto", "bfloat16", "bf16"):
        raise NotImplementedError(f"V4.1 cache planes are BF16; a {cache_dtype!r} KV cache is not implemented")
    if str(getattr(cache_config, "cache_dtype", "")).lower() != "bfloat16":
        logger.warning("V4.1 pin cache_dtype %s -> bfloat16 (DSV4F validate_cache_runtime)", cache_dtype)
    cache_config.cache_dtype = "bfloat16"


def plan_v41_layer_placements(specs):
    """Return {layer: (byte_offset, page_stride)} for every V4.1 plane.

    DSV4F plan_cache_slots: KV at 0, shared index at kv_bytes, aliases at 0,
    standalone index at 0 on its own page. Decode SFM/QLI must view the same
    addresses scatter_cache_sk wrote.
    """
    page_sizes, layer_tuples = get_layer_tuples(specs)
    placements = {}
    for page_size, layer_tuple in zip(page_sizes, layer_tuples):
        if len(layer_tuple) == 1 and getattr(specs[layer_tuple[0]], "scale_dim", None):
            placements[layer_tuple[0]] = (0, page_size)
            continue
        kv_name, *rest = layer_tuple
        kv_bytes = sum(_cache_plane_sizes(specs[kv_name]))
        placements[kv_name] = (0, page_size)
        if rest and getattr(specs[rest[0]], "scale_dim", 0):
            index_name, *rest = rest
            placements[index_name] = (kv_bytes, page_size)
        for name in rest:
            placements[name] = (0, page_size)
    return placements


def reshape_v41_cache(raw, spec, *, num_blocks, offset, block_stride):
    """Create typed per-page views using the containing slot's physical stride.

    DSV4F's reshape_cache: first stride is the physical page, not the logical
    plane. Decode SFM/QLI must see the same addresses scatter_cache_sk wrote.
    """
    if raw.dtype != torch.uint8:
        raw = raw.view(torch.uint8).reshape(-1)
    # Derived from tokens_per_state, not the optional MLA override field: a
    # compress_ratio-2 plane stores half a logical block per page, and reading
    # block_size rows instead would stride into the next page.
    from vllm_ascend.core.kv_cache_interface import get_storage_block_size

    storage_block_size = int(get_storage_block_size(spec))
    num_kv_heads = int(spec.num_kv_heads)
    head_size = int(spec.head_size)

    def view(dtype, width, byte_offset):
        dtype_size = dtype.itemsize
        storage_offset = int(raw.storage_offset()) + int(byte_offset)
        if storage_offset % dtype_size or block_stride % dtype_size:
            raise ValueError("V4.1 cache offset/stride is not dtype aligned")
        return torch.as_strided(
            raw.view(dtype),
            size=(num_blocks, storage_block_size, num_kv_heads, width),
            stride=(block_stride // dtype_size, num_kv_heads * width, width, 1),
            storage_offset=storage_offset // dtype_size,
        )

    key = view(spec.dtype, head_size, offset)
    scale_dim = int(getattr(spec, "scale_dim", 0) or 0)
    if scale_dim:
        key_bytes = storage_block_size * num_kv_heads * head_size * spec.dtype.itemsize
        return key, view(spec.scale_dtype, scale_dim, offset + key_bytes)
    return key
