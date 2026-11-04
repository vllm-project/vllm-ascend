# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp layerwise owner bindings over the original packed allocation."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseLayerCacheSpecs,
    LayerwiseReuseLayout,
    NamedKVCacheSpec,
    build_layerwise_cache_layout,
    get_layerwise_kv_cache_specs,
    get_layerwise_reuse_config,
    get_physical_layer_index,
)
from vllm_ascend.models.qwen4_exp.cache_config import (
    GDN,
    HIDDEN,
    PLE,
    QSA_COMPRESSED,
    QSA_MAIN,
    QSA_RAW,
    Qwen4KVCacheLayout,
    build_qwen4_kv_cache_layout,
    make_qsa_kernel_page_cache,
    make_slab_view,
)


@dataclass(frozen=True)
class Qwen4LayerwiseReuseLayout(LayerwiseReuseLayout):
    cache_slot_map: dict[str, int]


def build_qwen4_reuse_layout(
    layer_specs: dict[str, KVCacheSpec], base_layers: int, extra_config: dict[str, Any]
) -> Qwen4LayerwiseReuseLayout:
    """Rotate main/state owners while retaining auxiliary physical slots.

    QSA and GDN interpret the same packed r0 bytes using different views, so
    semantic spec equality is not the compatibility criterion for these slots.
    The model-owned layout validates page sizes/alignment before allocation.
    """
    by_layer: dict[int, list[NamedKVCacheSpec]] = {}
    for name, spec in layer_specs.items():
        by_layer.setdefault(get_physical_layer_index(name, base_layers), []).append(NamedKVCacheSpec(name, spec))
    physical_layers = sorted(by_layer)
    base = build_layerwise_cache_layout(len(physical_layers), extra_config)
    specs_by_layer: dict[int, LayerwiseLayerCacheSpecs] = {}
    qsa_layers: list[int] = []
    for layer, specs in by_layer.items():
        mains = [
            item
            for item in specs
            if (item.layer_name.endswith(".attn") and isinstance(item.spec, FullAttentionSpec))
            or (
                item.layer_name.endswith(".linear_attn")
                and isinstance(item.spec, MambaSpec)
                and len(item.spec.shapes) == 2
            )
        ]
        compressed = [item for item in specs if item.layer_name.endswith(".indexer.compressed_key_cache")]
        raw = [item for item in specs if item.layer_name.endswith(".indexer.raw_key_cache")]
        if len(mains) != 1:
            raise ValueError(f"Qwen layer {layer} requires exactly one QSA or GDN main cache")
        if isinstance(mains[0].spec, FullAttentionSpec):
            if (
                len(raw) != 1
                or len(compressed) != 1
                or not isinstance(raw[0].spec, CircularBufferSpec)
                or not isinstance(compressed[0].spec, MLAAttentionSpec)
            ):
                raise ValueError(f"Qwen layer {layer} requires raw and compressed indexer caches")
            qsa_layers.append(layer)
        elif raw or compressed:
            raise ValueError(f"GDN layer {layer} cannot own a QSA indexer")
        indexer = compressed[0] if compressed else None
        extras = tuple(item for item in specs if item is not mains[0] and item is not indexer)
        if any(
            not (item in raw or (isinstance(item.spec, MambaSpec) and len(item.spec.shapes) == 1)) for item in extras
        ):
            raise ValueError(f"Unsupported Qwen auxiliary cache in layer {layer}")
        specs_by_layer[layer] = LayerwiseLayerCacheSpecs(mains[0], indexer, extras)
    slots = tuple(tuple(physical_layers[i] for i in members) for members in base.storage_indices)
    slot_map: dict[str, int] = {}
    if base.has_layer_reuse:
        if len(slots) > len(qsa_layers):
            raise ValueError("Qwen reuse buffers plus independent layers exceed the existing packed slab count")
        for slot, members in enumerate(slots):
            for layer in members:
                slot_map[specs_by_layer[layer].main.layer_name] = slot
        for slot, layer in enumerate(sorted(qsa_layers)):
            for item in by_layer[layer]:
                if item.layer_name.endswith((".indexer.raw_key_cache", ".indexer.compressed_key_cache")):
                    slot_map[item.layer_name] = slot
    return Qwen4LayerwiseReuseLayout(
        layer_cache_specs=specs_by_layer,
        buffer_slots=slots,
        prefetch_layer_map={physical_layers[i]: physical_layers[j] for i, j in base.prefetch_layer_map.items()},
        independent_layers=[physical_layers[i] for i in base.independent_layers],
        num_prefetch_layers=base.num_prefetch_layers,
        has_layer_reuse=base.has_layer_reuse,
        cache_slot_map=slot_map,
    )


def get_qwen4_layerwise_slot_map(kv_cache_config: KVCacheConfig, vllm_config: VllmConfig) -> dict[str, int]:
    extra = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
    if extra is None:
        return {}
    specs = get_layerwise_kv_cache_specs(kv_cache_config)
    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    return build_qwen4_reuse_layout(specs, base_layers, extra).cache_slot_map


def apply_qwen4_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    excluded_layer_names: set[str] | None = None,
) -> Qwen4LayerwiseReuseLayout | None:
    """Validate packed bindings without rewriting the KVCacheTensor descriptors."""
    extra = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
    if extra is None or not kv_cache_config.kv_cache_tensors:
        return None
    specs = get_layerwise_kv_cache_specs(kv_cache_config)
    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    layout = build_qwen4_reuse_layout(specs, base_layers, extra)
    if not layout.has_layer_reuse:
        return None
    if (excluded_layer_names or set()) & specs.keys():
        raise ValueError("Qwen packed layerwise reuse does not support excluded draft cache owners")
    bind_qwen4_layerwise_cache_layout(
        build_qwen4_kv_cache_layout(kv_cache_config.kv_cache_groups, kv_cache_config.num_blocks),
        layout.cache_slot_map,
    )
    logger.info(
        "Qwen layerwise KV cache reuse binds %d layers to %d main/state slots; "
        "packed geometry and normal indexer offload are preserved.",
        len(layout.layer_cache_specs),
        len(layout.buffer_slots),
    )
    return layout


def bind_qwen4_layerwise_cache_layout(
    layout: Qwen4KVCacheLayout | None, cache_slot_map: dict[str, int]
) -> Qwen4KVCacheLayout:
    """Bind owners to reuse slots without changing the model's slab geometry."""
    if layout is None:
        raise ValueError("Qwen layerwise reuse requires a packed Qwen cache layout")
    owner_names = {owner.layer_name for owner in layout.owners}
    if not cache_slot_map.keys() <= owner_names:
        raise ValueError("Qwen reuse plan contains unknown cache owners")
    for owner in layout.owners:
        slot = cache_slot_map.get(owner.layer_name, owner.slot)
        if slot < 0 or slot >= layout.slot_count:
            raise ValueError(f"Qwen reuse slot {slot} exceeds the packed backing")
        if owner.role not in (QSA_MAIN, GDN) and slot != owner.slot:
            raise ValueError("Qwen auxiliary caches must retain their original slots")
    return replace(
        layout,
        owners=tuple(replace(owner, slot=cache_slot_map.get(owner.layer_name, owner.slot)) for owner in layout.owners),
    )


def allocate_qwen4_cache(
    layout: Qwen4KVCacheLayout, allocate: Callable[[int], torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Allocate the original packed backing, including every indexer region."""
    backing = allocate(layout.slot_count * layout.slot_backing_size)
    caches = {}
    for owner in layout.owners:
        if owner.role in (PLE, HIDDEN):
            size = (
                owner.spec.state_content_size_bytes if isinstance(owner.spec, MambaSpec) else owner.spec.page_size_bytes
            )
            caches[owner.layer_name] = allocate(layout.num_blocks * size)
        else:
            caches[owner.layer_name] = backing
    return caches


def view_qwen4_cache(
    layout: Qwen4KVCacheLayout,
    layer_name: str,
    raw: torch.Tensor,
    kernel_block_size: int,
) -> torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None:
    """Bind role-specific views without changing packed page geometry."""
    owner = layout.owner(layer_name)
    spec = owner.spec
    slot_offset = owner.slot * layout.slot_backing_size
    if owner.role == QSA_MAIN:
        assert isinstance(spec, FullAttentionSpec)
        region = layout.region("r0")
        return make_qsa_kernel_page_cache(
            raw,
            dtype=spec.dtype,
            num_blocks=layout.num_blocks,
            manager_block_size=spec.block_size,
            kernel_block_size=kernel_block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            storage_offset=slot_offset + region.offset,
            block_stride=region.block_stride,
        )
    if owner.role in (QSA_RAW, QSA_COMPRESSED):
        assert isinstance(spec, (CircularBufferSpec, MLAAttentionSpec))
        region = layout.region("r4" if owner.role == QSA_RAW else "r5")
        block_size = spec.block_size if owner.role == QSA_RAW else spec.num_states
        if block_size * spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype) != region.page_size_bytes:
            raise ValueError(f"{layer_name} Qwen indexer page does not match its packed region")
        num_blocks = layout.num_blocks
        if owner.role == QSA_COMPRESSED:
            num_blocks *= spec.block_size // kernel_block_size
            block_size = spec.get_num_kernel_states(kernel_block_size)
        return make_slab_view(
            raw,
            dtype=spec.dtype,
            num_blocks=num_blocks,
            item_shape=(spec.num_kv_heads, block_size, spec.head_size),
            storage_offset=slot_offset + region.offset,
        ).transpose(1, 2)
    if owner.role == PLE:
        assert isinstance(spec, MambaSpec)
        return [
            make_slab_view(
                raw,
                dtype=spec.dtypes[0],
                num_blocks=layout.num_blocks,
                item_shape=tuple(spec.shapes[0]),
                storage_offset=0,
            )
        ]
    if owner.role == GDN:
        assert isinstance(spec, MambaSpec)
        return [
            make_slab_view(
                raw,
                dtype=dtype,
                num_blocks=layout.num_blocks,
                item_shape=tuple(shape),
                storage_offset=slot_offset + layout.region(region).offset,
                block_stride=layout.region(region).block_stride,
            )
            for shape, dtype, region in zip(spec.shapes, spec.dtypes, ("r1", "r2"), strict=True)
        ]
    if owner.role != HIDDEN:
        raise ValueError(f"Unsupported Qwen cache owner {owner}")
    return None
