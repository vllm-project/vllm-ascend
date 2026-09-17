# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 layer-outermost cache placement and allocation."""

from dataclasses import replace

from vllm.config import VllmConfig
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

STATE_RING_ROWS = 32


def is_deepseek_v41_cache(specs_or_groups):
    if isinstance(specs_or_groups, dict):
        specs = specs_or_groups.values()
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
        kv_bytes = kv_spec.unpadded_page_size_bytes
        index_bytes = index_spec.unpadded_page_size_bytes
        if slot_idx < len(draft_swa):
            aliases.append(draft_swa[slot_idx])
        capacity = max(
            kv_bytes + index_bytes,
            *(specs[name].unpadded_page_size_bytes for name in aliases),
        )
        layer_tuples.append((kv_name, index_name, *aliases))
        page_sizes.append(capacity)
    return page_sizes, layer_tuples


def group_cache_specs(specs):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    page_sizes, layer_tuples = get_layer_tuples(specs)
    padded = {}
    for page_size, layer_tuple in zip(page_sizes, layer_tuples):
        kv_name, index_name, *aliases = layer_tuple
        kv_bytes = specs[kv_name].unpadded_page_size_bytes
        padded[kv_name] = replace(specs[kv_name], page_size_padded=kv_bytes)
        padded[index_name] = replace(specs[index_name], page_size_padded=page_size - kv_bytes)
        padded.update((name, replace(specs[name], page_size_padded=page_size)) for name in aliases)

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


def get_deepseek_v41_kv_cache_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Allocate four independent layer slots backed by one global block-ID pool."""
    page_sizes, layer_tuples = get_layer_tuples(_specs_from_groups(groups))
    capacity = max(available_memory // sum(page_sizes), 0)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
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
