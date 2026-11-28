# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache placements within a shared allocation."""

from dataclasses import dataclass, replace
from itertools import chain, zip_longest

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

# Both supported versions expose planning helpers in kv_cache_planning.
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
    declared_kwarg,
)
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    get_current_hardware_profile,
)
from vllm_ascend.models.deepseek_v41.long_kv import LONG_KV_INT8_GROUP

LAYER_PACKED_SLOTS = 2
STATE_RING_ROWS = 32

# Fixed dimensions accepted by the packaged mixed-quant attention ABI.
# The actual head sizes come from the model config; these bounds reject
# unsupported layouts before allocating caches, rather than override config.
MLA_HEAD_DIM = 512
INDEX_HEAD_DIM = 128
INDEX_FOLD_ROWS = 8


def uses_a5_packed_cache() -> bool:
    """Select the V4.1 cache ABI from the detected hardware profile."""
    return get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE)


def make_swa_cache_spec(*, block_size, window_size, head_size, dtype, cache_dtype, bounded_replay=False):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # FP8 values followed by one BF16 scale per 32 values.
        head_size += (head_size // 32) * torch.bfloat16.itemsize
        dtype = torch.uint8
        cache_dtype = "a5_mxfp8_bf16_scale"
    return AscendSlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        sliding_window=window_size,
        cache_dtype_str=cache_dtype,
        model_version="deepseek_v41",
        alignment=None,
        **declared_kwarg(AscendSlidingWindowMLASpec, "bounded_replay", bounded_replay),
    )


def make_mla_cache_spec(*, block_size, head_size, compress_ratio, int8_long_kv=False):
    if int8_long_kv:
        if uses_a5_packed_cache() or head_size != MLA_HEAD_DIM:
            raise ValueError("INT8 G64 long KV requires A2/A3 and a 512-channel V4.1 head")
        head_size += (head_size // LONG_KV_INT8_GROUP) * torch.float32.itemsize
        dtype = torch.int8
        # The FP32 scales occupy the last 32 bytes of the same row.
        scale_dtype = torch.float32
    elif uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # Two FP4 values per byte, followed by one BF16 scale per 16 values.
        head_size = head_size // 2 + (head_size // 16) * torch.bfloat16.itemsize
        dtype = torch.uint8
        scale_dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
        scale_dtype = torch.int8
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        scale_dtype=scale_dtype,
        layer_packed=int8_long_kv,
    )


def make_index_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != INDEX_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {INDEX_HEAD_DIM}, got {head_size}")
        scale_dim = head_size // 32
        head_size //= 2
        dtype = torch.uint8
        scale_dtype = torch.uint8
    else:
        dtype = torch.int8
        scale_dim = 1
        scale_dtype = torch.float16
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
    )


def make_folded_index_cache_spec(*, block_size):
    """QSLI candidate-source view: eight K/scale rows per 544-byte row."""
    if block_size % INDEX_FOLD_ROWS:
        raise ValueError("A5 folded index page must contain full 8-token groups")
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=INDEX_FOLD_ROWS * (INDEX_HEAD_DIM // 2 + INDEX_HEAD_DIM // 32),
        dtype=torch.uint8,
        tokens_per_state=INDEX_FOLD_ROWS,
        model_version="deepseek_v41",
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
    return any(getattr(spec, "model_version", None) == "deepseek_v41" for spec in specs)


def _layer_number(name):
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def _draft_layer_number(name):
    return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])


def get_layer_tuples(specs):
    """Return DSV4-style ordered layer tuples and their physical page sizes."""
    mla = {name for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec)}
    state = sorted((name for name, spec in specs.items() if isinstance(spec, CircularBufferSpec)), key=_layer_number)
    swa = {name for name, spec in specs.items() if isinstance(spec, AscendSlidingWindowMLASpec)}

    full = sorted((name for name in mla if name.endswith(".long_kv_cache")), key=_layer_number)
    target_swa = sorted((name for name in swa if ".mtp." not in f".{name}"), key=_layer_number)
    draft_swa = sorted((name for name in swa if ".mtp." in f".{name}"), key=_draft_layer_number)

    # A reduced backbone may contain only the initial SWA layers. Without
    # full-context source slots, each SWA layer needs its own physical slot.
    if not full and len(swa) == len(specs):
        names = target_swa + draft_swa
        return [specs[name].unpadded_page_size_bytes for name in names], [(name,) for name in names]

    layer_tuples: list[tuple[str, ...]] = []
    page_sizes: list[int] = []
    for slot_idx, kv_name in enumerate(full):
        prefix = kv_name.rsplit(".", 1)[0]
        index_name = prefix + ".indexer.k_cache"
        folded_name = index_name + "_folded"
        folded_spec = specs.get(folded_name)
        index_spec = specs[index_name]
        kv_spec = specs[kv_name]
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + target_swa[slot_idx :: len(full)]
        kv_bytes = kv_spec.unpadded_page_size_bytes
        index_bytes = index_spec.unpadded_page_size_bytes
        folded_bytes = folded_spec.unpadded_page_size_bytes if folded_spec is not None else 0
        if slot_idx < len(draft_swa):
            aliases.append(draft_swa[slot_idx])
        capacity = max(
            kv_bytes + index_bytes + folded_bytes,
            *(specs[name].unpadded_page_size_bytes for name in aliases),
        )
        layer_tuples.append((kv_name, index_name, *((folded_name,) if folded_spec is not None else ()), *aliases))
        page_sizes.append(capacity)
    return page_sizes, layer_tuples


@dataclass(kw_only=True)
class LayerPackedKVCacheConfig(KVCacheConfig):
    # Each owner starts at byte zero of its physical slot.
    page_owners: tuple[str, ...]


def dummy_ring_pages_per_group(num_blocks: int, num_reqs: int, num_ring_groups: int) -> int:
    """Budget disjoint writable IDs across packed ring groups, reserving block 0."""
    if num_blocks < 1:
        raise ValueError("Layer-packed dummy attention requires at least the null block")
    return min(num_reqs, (num_blocks - 1) // num_ring_groups)


def _is_layer_packed(specs):
    return any(isinstance(spec, AscendMLAAttentionSpec) and spec.layer_packed for spec in specs.values())


def _long_index_pairs(specs):
    sources = sorted((n for n in specs if n.endswith(".long_kv_cache")), key=_layer_number)
    return [(src, src.rsplit(".", 1)[0] + ".indexer.k_cache") for src in sources]


def _layer_packed_groups(specs):
    rings = sorted((n for n, s in specs.items() if isinstance(s, CircularBufferSpec)), key=_layer_number)
    swa = [n for n, s in specs.items() if isinstance(s, AscendSlidingWindowMLASpec)]
    target = sorted((n for n in swa if ".mtp." not in f".{n}"), key=_layer_number)
    draft = sorted((n for n in swa if ".mtp." in f".{n}"), key=_draft_layer_number)
    # Preserve the long-KV source as the representative spec for the scheduler.
    groups = [list(chain.from_iterable(_long_index_pairs(specs)))]
    for names in (rings, target, draft):
        groups.extend(names[i : i + LAYER_PACKED_SLOTS] for i in range(0, len(names), LAYER_PACKED_SLOTS))
    return [UniformTypeKVCacheSpecs.from_specs({n: specs[n] for n in names}) for names in groups]


def _layer_packed_slots(specs, groups):
    # Keep each source beside its index. Place the largest pair first so one
    # C1 and three C2 sources fit below the two full-size SWA pages.
    pairs = _long_index_pairs(specs)

    def pair_bytes(pair):
        return sum(specs[n].unpadded_page_size_bytes for n in pair)

    slots = [[] for _ in range(LAYER_PACKED_SLOTS)]
    sizes = [0] * LAYER_PACKED_SLOTS
    for pair in sorted(pairs, key=lambda p: (-pair_bytes(p), _layer_number(p[0]))):
        slot = min(range(LAYER_PACKED_SLOTS), key=sizes.__getitem__)
        slots[slot].extend(pair)
        sizes[slot] += pair_bytes(pair)
    for group in groups:
        if isinstance(group.kv_cache_spec.first_spec, AscendMLAAttentionSpec):
            continue
        for slot, name in enumerate(group.layer_names):
            slots[slot].append(name)
            sizes[slot] = max(sizes[slot], specs[name].unpadded_page_size_bytes)
    return [size for size in sizes if size], [slot for slot in slots if slot]


def group_cache_specs(specs):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    if _is_layer_packed(specs):
        return _layer_packed_groups(specs)
    if specs and all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs.values()):
        return [UniformTypeKVCacheSpecs.from_specs({name: spec}) for name, spec in specs.items()]
    page_sizes, layer_tuples = get_layer_tuples(specs)
    padded = {}
    for page_size, layer_tuple in zip(page_sizes, layer_tuples):
        kv_name, index_name, *rest = layer_tuple
        folded_name = index_name + "_folded"
        has_folded = folded_name in rest
        aliases = rest[1:] if has_folded else rest
        kv_bytes = specs[kv_name].unpadded_page_size_bytes
        index_bytes = specs[index_name].unpadded_page_size_bytes
        padded[kv_name] = replace(specs[kv_name], page_size_padded=kv_bytes)
        padded[index_name] = replace(
            specs[index_name], page_size_padded=index_bytes if has_folded else page_size - kv_bytes
        )
        if has_folded:
            padded[folded_name] = replace(specs[folded_name], page_size_padded=page_size - kv_bytes - index_bytes)
        padded.update((name, replace(specs[name], page_size_padded=page_size)) for name in aliases)

    mla_names = [name for name, spec in padded.items() if isinstance(spec, AscendMLAAttentionSpec)]
    state_names = [name for name, spec in padded.items() if isinstance(spec, CircularBufferSpec)]
    # A decoder-only PP stage has no compression state buffers.
    groups = [
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in names})
        for names in (mla_names, state_names)
        if names
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
    ]
    # Stage-local SWA counts need not be divisible by the number of KV sources.
    for names in zip_longest(*swa_columns):
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in names if name is not None}))

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
            group_spec = group.kv_cache_spec
            specs[name] = (
                group_spec.kv_cache_specs[name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            )
    return specs


def _get_worker_layer_tuples(groups):
    """Pack scheduler groups whose width may exceed the stage's KV source count."""
    specs = _specs_from_groups(groups)
    if _is_layer_packed(specs):
        return _layer_packed_slots(specs, groups)
    if specs and all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs.values()):
        # Preserve main's independent slots for SWA-only backbones and drafts.
        _, slots = get_layer_tuples(specs)
        return [max(specs[name].page_size_bytes for name in names) for names in slots], slots
    _, source_tuples = get_layer_tuples(specs)
    # A column's entries need distinct slots; different columns may alias.
    columns = [
        [tuple(name for name in names if isinstance(specs[name], AscendMLAAttentionSpec)) for names in source_tuples]
    ]
    columns.extend(
        [(name,) for name in group.layer_names if not isinstance(specs[name], AscendMLAAttentionSpec)]
        for group in groups
    )
    slots = [tuple(chain.from_iterable(parts)) for parts in zip_longest(*columns, fillvalue=())]
    page_sizes = [
        max(
            sum(specs[name].page_size_bytes for name in names if isinstance(specs[name], AscendMLAAttentionSpec)),
            *(specs[name].page_size_bytes for name in names),
        )
        for names in slots
    ]
    return page_sizes, slots


def get_deepseek_v41_pool_bytes_per_block(groups):
    page_sizes, _ = _get_worker_layer_tuples(groups)
    return sum(page_sizes)


def get_deepseek_v41_kv_cache_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    num_blocks: int,
) -> KVCacheConfig:
    """Place stage-local resource pages within one shared backing pool."""
    specs = _specs_from_groups(groups)
    page_sizes, layer_tuples = _get_worker_layer_tuples(groups)
    backing_size = num_blocks * sum(page_sizes)
    tensors: list[KVCacheTensor] = []
    slot_offset = 0
    for page_size, layer_names in zip(page_sizes, layer_tuples):
        # Long KV, split indexer and optional folded indexer are consecutive
        # within a page. State/SWA groups reuse that page from its beginning.
        component_offset = 0
        for name in layer_names:
            spec = specs[name]
            is_mla = isinstance(spec, AscendMLAAttentionSpec)
            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=[name],
                    offset=slot_offset + (component_offset if is_mla else 0),
                    layer_stride=num_blocks * page_size,
                    block_stride=page_size,
                )
            )
            if is_mla:
                component_offset += spec.unpadded_page_size_bytes
        slot_offset += num_blocks * page_size
    fields = dict(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
    if _is_layer_packed(specs):
        logger.info(
            "V4.1 layer-packed KV: %d shared blocks, %s bytes per slot, %d groups", num_blocks, page_sizes, len(groups)
        )
        return LayerPackedKVCacheConfig(**fields, page_owners=tuple(names[0] for names in layer_tuples))
    return KVCacheConfig(**fields)
