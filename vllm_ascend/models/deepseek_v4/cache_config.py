# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 KV cache placement and allocation layout."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec, KVCacheTensor, UniformTypeKVCacheSpecs


@dataclass(frozen=True)
class DeepSeekV4KVCacheLayout:
    """Shared backing and per-layer regions emitted by the DSV4 planner."""

    backing_size: int
    regions: tuple[tuple[str, int, int], ...]
    num_blocks: int


def _group_member_specs(group: KVCacheGroupSpec) -> dict[str, KVCacheSpec]:
    if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
        return group.kv_cache_spec.kv_cache_specs
    return {name: group.kv_cache_spec for name in group.layer_names}


def _descriptor_layers(descriptor: KVCacheTensor) -> tuple[str, ...]:
    layers = getattr(descriptor, "layers", None)
    if layers is None:
        layers = getattr(descriptor, "shared_by", None)
    return tuple(layers or ())


def build_dsv4_kv_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
    kv_cache_tensors: list[KVCacheTensor],
    num_blocks: int,
    *,
    uses_turboquant: bool,
) -> DeepSeekV4KVCacheLayout | None:
    """Build the shared-tuple DSV4 cache layout from planner descriptors.

    Returning ``None`` means the cache groups are not DSV4. Once a DSV4 spec
    is present, malformed descriptor geometry raises immediately so the
    allocator cannot fall through to a generic layout.
    """
    layer_specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_groups:
        layer_specs.update(_group_member_specs(group))

    if not any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in layer_specs.values()):
        return None
    if num_blocks <= 0:
        raise ValueError("DeepSeek-V4 KV cache layout requires a positive block count.")
    if not kv_cache_tensors:
        raise ValueError("DeepSeek-V4 KV cache config has no descriptors.")

    sizes = {descriptor.size for descriptor in kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("DeepSeek-V4 KV cache descriptors must share one backing allocation.")
    backing_size = sizes.pop()
    regions: list[tuple[str, int, int]] = []
    for descriptor in kv_cache_tensors:
        names = _descriptor_layers(descriptor)
        if not names:
            raise ValueError("DeepSeek-V4 KV cache descriptors must name at least one layer.")
        for layer_idx, layer_name in enumerate(names):
            if layer_name not in layer_specs:
                raise ValueError(f"DeepSeek-V4 descriptor references unknown cache layer {layer_name}.")
            spec = layer_specs[layer_name]
            if not uses_turboquant and descriptor.block_stride != spec.page_size_bytes:
                raise ValueError(
                    "DeepSeek-V4 requires contiguous per-layer pages, "
                    f"but {layer_name} has block_stride={descriptor.block_stride} "
                    f"and page_size={spec.page_size_bytes}."
                )
            if descriptor.block_stride < spec.page_size_bytes:
                raise ValueError(f"DSA physical stride is smaller than the page for {layer_name}")
            layer_size = (num_blocks - 1) * descriptor.block_stride + spec.page_size_bytes
            start = descriptor.offset + layer_idx * descriptor.layer_stride
            if start < 0 or start + layer_size > backing_size:
                raise ValueError(f"DeepSeek-V4 KV cache view for {layer_name} exceeds the shared backing allocation.")
            regions.append((layer_name, start, layer_size))

    if not regions:
        raise ValueError("DeepSeek-V4 KV cache config has no materializable layers.")
    return DeepSeekV4KVCacheLayout(backing_size, tuple(regions), num_blocks)
