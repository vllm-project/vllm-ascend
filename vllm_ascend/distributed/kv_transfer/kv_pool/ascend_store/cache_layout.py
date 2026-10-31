# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Describe registered cache components without model-name dispatch.

Local addresses never enter wire descriptions. Semantic token/head axes are
selected by the cache-spec adapter; shape alone is not proof of cache semantics.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, SlidingWindowMLASpec, SlidingWindowSpec

from vllm_ascend.core.kv_cache_interface import get_storage_block_size, is_circular_kv_cache_spec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import CacheSlice, TargetCacheSlice

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

MLA_PAGE_FORMAT_PREFIX = "mla-page-v2:"


@dataclass(frozen=True)
class CacheComponentBinding:
    layer: str
    component: int
    group_id: int
    spec_type: str
    dtype: str
    address: int
    shape: tuple[int, ...]
    byte_strides: tuple[int, ...]
    element_size: int
    scheduler_blocks: int

    @property
    def block_size_scale(self) -> int:
        return self.shape[0] // self.scheduler_blocks

    @property
    def page_span(self) -> int:
        return self.element_size + sum(
            (size - 1) * stride for size, stride in zip(self.shape[1:], self.byte_strides[1:])
        )

    def to_wire(self) -> dict[str, Any]:
        """Physical component description; deliberately omit local HBM addresses."""
        return {
            "layer": self.layer,
            "component": self.component,
            "group_id": self.group_id,
            "spec_type": self.spec_type,
            "dtype": self.dtype,
            "shape": list(self.shape[1:]),
            "byte_strides": list(self.byte_strides[1:]),
            "block_stride": self.byte_strides[0],
            "block_size_scale": self.block_size_scale,
            "element_size": self.element_size,
        }

    def token_page(
        self,
        physical_block_id: int,
        *,
        token_start: int,
        head_start: int,
        token_axis: int,
        head_axis: int | None,
        format_id: str,
    ) -> TargetCacheSlice:
        """Bind one kernel page after its logical CP/TP placement is known.

        Axes index the original tensor, including the physical block axis 0.
        Remaining trailing dimensions must form one contiguous byte vector.
        Recurrent state and opaque packed pages need their own validated adapter.
        """
        if type(physical_block_id) is not int or not 0 <= physical_block_id < self.shape[0]:
            raise ValueError("Physical cache block is out of bounds")
        axes = {token_axis} if head_axis is None else {token_axis, head_axis}
        if token_axis == 0 or any(type(axis) is not int or not 0 < axis < len(self.shape) for axis in axes):
            raise ValueError("Token/head axes must be non-block tensor axes")
        if head_axis is not None and token_axis == head_axis:
            raise ValueError("Token and head axes must be distinct")
        feature_axes = [axis for axis in range(1, len(self.shape)) if axis not in axes]
        width = self.element_size
        for axis in reversed(feature_axes):
            if self.shape[axis] > 1 and self.byte_strides[axis] != width:
                raise ValueError("Cache feature vector is not byte-contiguous")
            width *= self.shape[axis]
        tokens = self.shape[token_axis]
        heads = self.shape[head_axis] if head_axis is not None else 1
        cache = CacheSlice(
            self.layer,
            str(self.component),
            format_id,
            token_start,
            token_start + tokens,
            head_start,
            head_start + heads,
            width,
            self.byte_strides[token_axis],
            self.byte_strides[head_axis] if head_axis is not None else tokens * self.byte_strides[token_axis],
        )
        return TargetCacheSlice(cache, self.address + physical_block_id * self.byte_strides[0])


def bind_token_page(
    binding: CacheComponentBinding,
    spec: Any,
    physical_block_id: int,
    *,
    token_start: int,
    head_start: int = 0,
    storage_rows: int | None = None,
) -> TargetCacheSlice:
    """Interpret existing attention specs, including MLA/indexer components.

    Logical placement (TP heads and CP token stripes) is supplied by the caller.
    Recurrent/circular state is deliberately kept out of the token-KV path.
    """
    if is_circular_kv_cache_spec(spec):
        raise ValueError("Circular state requires a state-snapshot adapter")
    rows = get_storage_block_size(spec) if storage_rows is None else storage_rows
    if type(rows) is not int or rows <= 0:
        raise ValueError("Cache component requires positive physical token rows")
    shape = binding.shape
    mla = isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))
    opaque_page = mla and any(
        bool(getattr(spec, name, False))
        for name in (
            "cache_sparse_sfa_c8",
            "cache_sparse_sfa_turboquant",
            "cache_sparse_li_c8",
            "cache_sparse_li_c4",
        )
    )
    if mla:
        if not opaque_page and (len(shape) < 3 or shape[1] != rows):
            raise ValueError("MLA component requires an explicit physical token-row layout")
        token_axis, head_axis = 1, None
        if head_start != 0:
            raise ValueError("MLA latent/indexer components use one logical cache head")
    elif isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
        heads = spec.num_kv_heads
        if len(shape) != 4:
            raise ValueError("Attention KV components require block/head/token/feature dimensions")
        hnd = shape[1] == heads and shape[2] == rows
        nhd = shape[1] == rows and shape[2] == heads
        if hnd and nhd and heads > 1:
            raise ValueError("Ambiguous HND/NHD component requires an explicit layout")
        if hnd:
            token_axis, head_axis = 2, 1
        elif nhd:
            token_axis, head_axis = 1, 2
        else:
            raise ValueError("Attention component dimensions disagree with its cache spec")
    else:
        raise ValueError(f"No token-cache interpretation for {type(spec).__name__}")
    # Include semantic byte-format fields, but not topology, page size, group,
    # or speculative step count. Those may legitimately differ across P/D.
    signature = {
        "family": "mla" if mla else "attention",
        "dtype": binding.dtype,
        "component": binding.component,
    }
    for name in (
        "cache_dtype_str",
        "cache_sparse_sfa_c8",
        "cache_sparse_sfa_turboquant",
        "cache_sparse_li_c8",
        "cache_sparse_li_c4",
        "scale_dim",
        "scale_dtype",
        "model_version",
        "tokens_per_state",
    ):
        value = getattr(spec, name, None)
        if value is not None:
            signature[name] = str(value)
    if opaque_page:
        # Match V2's whole-kernel-page path. Opaque C8/TurboQuant contents
        # cannot be interpreted as a constant-stride token array.
        if token_start % rows or not 0 <= physical_block_id < shape[0]:
            raise ValueError("Packed MLA requires aligned, valid physical pages")
        expected_stride = binding.element_size
        for size, stride in reversed(list(zip(shape[1:], binding.byte_strides[1:], strict=True))):
            if size > 1 and stride != expected_stride:
                raise ValueError("Packed MLA page must be byte-contiguous")
            expected_stride *= size
        if len(shape) > 2 and shape[1] != rows:
            raise ValueError("Packed MLA storage view disagrees with physical kernel rows")
        signature.update(family="mla-page-v2", rows=rows, shape=shape[1:])
        span = binding.page_span
        page = token_start // rows
        cache = CacheSlice(
            binding.layer,
            str(binding.component),
            MLA_PAGE_FORMAT_PREFIX + json.dumps(signature, sort_keys=True, separators=(",", ":")),
            page,
            page + 1,
            0,
            1,
            span,
            span,
            span,
        )
        return TargetCacheSlice(cache, binding.address + physical_block_id * binding.byte_strides[0])
    return binding.token_page(
        physical_block_id,
        token_start=token_start,
        head_start=head_start,
        token_axis=token_axis,
        head_axis=head_axis,
        format_id=json.dumps(signature, sort_keys=True, separators=(",", ":")),
    )


def describe_cache_bindings(
    config: "KVCacheConfig",
    registered: dict[str, Any],
    *,
    owned_layers: set[str] | None = None,
) -> dict[str, tuple[CacheComponentBinding, ...]]:
    """Describe every configured component, preserving draft layers and groups.

    KVPP ownership is computed with the existing placement helper by the caller.
    Unowned layers can remain registered as scratch but must not be exported.
    """
    if type(config.num_blocks) is not int or config.num_blocks <= 0:
        raise ValueError("Cache registration requires a positive scheduler block count")
    specs = get_layerwise_kv_cache_specs(config)
    groups: dict[str, int] = {}
    for index, group in enumerate(config.kv_cache_groups):
        for layer in group.layer_names:
            if layer in groups:
                raise ValueError(f"Cache layer belongs to multiple groups: {layer}")
            groups[layer] = index
    required = specs.keys() if owned_layers is None else owned_layers
    if registered.keys() - specs.keys() or required - registered.keys():
        raise ValueError(
            f"Registered cache layers differ from specs: missing={required - registered.keys()}, "
            f"extra={registered.keys() - specs.keys()}"
        )
    if owned_layers is not None and owned_layers - specs.keys():
        raise ValueError("Owned cache layers are absent from registered specs")
    result: dict[str, tuple[CacheComponentBinding, ...]] = {}
    for layer, value in registered.items():
        if owned_layers is not None and layer not in owned_layers:
            continue
        tensors = tuple(value) if isinstance(value, (tuple, list)) else (value,)
        if not tensors:
            raise ValueError(f"Cache layer has no registered components: {layer}")
        components = []
        for index, tensor in enumerate(tensors):
            shape = tuple(tensor.shape)
            # MLA without RoPE still registers the second component as a
            # zero-width view. It has no bytes to publish or restore. Keep the
            # latent component's identity, and reject other empty layouts.
            if (
                isinstance(specs[layer], (MLAAttentionSpec, SlidingWindowMLASpec))
                and len(tensors) == 2
                and index == 1
                and len(shape) >= 2
                and shape[-1] == 0
                and shape[:-1] == tuple(tensors[0].shape[:-1])
            ):
                continue
            element_size = tensor.element_size()
            strides = tuple(step * element_size for step in tensor.stride())
            if len(shape) < 2 or any(type(size) is not int or size <= 0 for size in shape):
                raise ValueError(f"Cache tensor has invalid dimensions: {layer}/{index}")
            if any(step <= 0 for step in strides) or shape[0] % config.num_blocks:
                raise ValueError(f"Cache tensor has invalid strides or physical block count: {layer}/{index}")
            span = element_size + sum((size - 1) * step for size, step in zip(shape, strides))
            storage = tensor.untyped_storage()
            address = tensor.data_ptr()
            if address < storage.data_ptr() or address + span > storage.data_ptr() + storage.nbytes():
                raise ValueError(f"Cache tensor view exceeds its backing storage: {layer}/{index}")
            spec_type = f"{type(specs[layer]).__module__}.{type(specs[layer]).__qualname__}"
            components.append(
                CacheComponentBinding(
                    layer,
                    index,
                    groups[layer],
                    spec_type,
                    str(tensor.dtype),
                    address,
                    shape,
                    strides,
                    element_size,
                    config.num_blocks,
                )
            )
        result[layer] = tuple(components)
    return result
