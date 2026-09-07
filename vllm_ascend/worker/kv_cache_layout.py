# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Allocation and view construction for explicit Ascend KV cache layouts."""

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendIndependentKVCacheTensor,
    AscendMLAAttentionSpec,
)

RawCache = torch.Tensor | tuple[torch.Tensor, ...]


def has_independent_kv_cache_tensors(kv_cache_config: KVCacheConfig) -> bool:
    return any(
        isinstance(descriptor, AscendIndependentKVCacheTensor) for descriptor in kv_cache_config.kv_cache_tensors
    )


def allocate_independent_kv_cache_tensors(
    kv_cache_config: KVCacheConfig,
    runner_only_layers: set[str],
    allocate_raw_tensor: Callable[[int], torch.Tensor],
) -> dict[str, RawCache]:
    """Allocate one contiguous backing tensor per physical descriptor."""
    descriptors = [
        descriptor
        for descriptor in kv_cache_config.kv_cache_tensors
        if isinstance(descriptor, AscendIndependentKVCacheTensor)
    ]
    if len(descriptors) != len(kv_cache_config.kv_cache_tensors):
        raise ValueError("Independent and shared-backing KV cache descriptors cannot be mixed in one cache plan.")

    raw_tensors: dict[str, RawCache] = {}
    for descriptor in descriptors:
        if descriptor.block_stride <= 0:
            raise ValueError(
                f"An independent cache descriptor requires a positive block stride, got {descriptor.block_stride}."
            )
        if descriptor.size != kv_cache_config.num_blocks * descriptor.block_stride:
            raise ValueError(
                "Independent cache descriptor size does not match its "
                "contiguous slot allocation: "
                f"size={descriptor.size}, "
                f"num_blocks={kv_cache_config.num_blocks}, "
                f"block_stride={descriptor.block_stride}."
            )
        if descriptor.offset != 0:
            raise ValueError(f"Independent cache descriptors must start at offset zero, got {descriptor.offset}.")

        backing = allocate_raw_tensor(descriptor.size)
        for layer_name in descriptor.shared_by:
            if layer_name in runner_only_layers:
                continue
            if layer_name in raw_tensors:
                raise ValueError(f"Cache layer {layer_name!r} has multiple independent physical descriptors.")
            raw_tensors[layer_name] = backing

    expected_layers = {
        layer_name
        for group in kv_cache_config.kv_cache_groups
        for layer_name in group.layer_names
        if layer_name not in runner_only_layers
    }
    allocated_layers = set(raw_tensors)
    if expected_layers != allocated_layers:
        raise AssertionError(
            "Independent KV cache tensors are not correctly initialized: "
            f"missing={sorted(expected_layers - allocated_layers)}, "
            f"unexpected={sorted(allocated_layers - expected_layers)}."
        )
    return raw_tensors


def _reshape_block_strided_tensor(
    raw_tensor: torch.Tensor,
    cache_shape: tuple[int, ...],
    dtype: torch.dtype,
    block_stride_bytes: int,
    block_offset_bytes: int,
    payload_offset_bytes: int = 0,
) -> torch.Tensor:
    if not cache_shape or cache_shape[0] <= 0:
        raise ValueError(f"Invalid independent cache shape: {cache_shape}.")

    dtype_size = get_dtype_size(dtype)
    storage_offset_bytes = block_offset_bytes + payload_offset_bytes
    if block_stride_bytes % dtype_size or storage_offset_bytes % dtype_size:
        raise ValueError(
            "Independent cache stride and offset must be aligned to "
            f"{dtype}: block_stride={block_stride_bytes}, "
            f"storage_offset={storage_offset_bytes}."
        )

    block_stride = block_stride_bytes // dtype_size
    payload_numel = math.prod(cache_shape[1:])
    block_end = storage_offset_bytes // dtype_size + payload_numel
    if block_end > block_stride:
        raise ValueError(
            "Independent cache payload crosses a physical block boundary: "
            f"shape={cache_shape}, block_stride={block_stride_bytes}, "
            f"block_offset={block_offset_bytes}, "
            f"payload_offset={payload_offset_bytes}."
        )

    typed_tensor = raw_tensor.view(dtype)
    relative_offset = storage_offset_bytes // dtype_size
    required_numel = relative_offset + (cache_shape[0] - 1) * block_stride + payload_numel
    if required_numel > typed_tensor.numel():
        raise ValueError(
            "Independent backing tensor is too small for its cache view: "
            f"required={required_numel}, available={typed_tensor.numel()}."
        )

    inner_strides = torch.empty(cache_shape[1:]).stride()
    return torch.as_strided(
        typed_tensor,
        size=cache_shape,
        stride=(block_stride, *inner_strides),
        storage_offset=typed_tensor.storage_offset() + relative_offset,
    )


def _get_cache_shape(
    backend: Any,
    num_blocks: int,
    block_size: int,
    spec: AttentionSpec,
) -> tuple[int, ...]:
    try:
        return backend.get_kv_cache_shape(
            num_blocks,
            block_size,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=getattr(spec, "cache_dtype_str", "auto") or "auto",
        )
    except TypeError:
        return backend.get_kv_cache_shape(
            num_blocks,
            block_size,
            spec.num_kv_heads,
            spec.head_size,
        )


def reshape_independent_kv_cache_tensors(
    kv_cache_config: KVCacheConfig,
    raw_tensors: dict[str, RawCache],
    layer_specs: dict[str, KVCacheSpec],
    attention_groups: Iterable[Any],
    runner_only_layers: set[str],
    get_attention_cache_dims: Callable[[str, AttentionSpec], tuple[int, int]],
) -> dict[str, Any]:
    """Build kernel-facing views from independent physical cache tensors."""
    descriptors: dict[str, AscendIndependentKVCacheTensor] = {}
    for descriptor in kv_cache_config.kv_cache_tensors:
        if not isinstance(descriptor, AscendIndependentKVCacheTensor):
            raise ValueError("Independent and shared-backing KV cache descriptors cannot be mixed in one cache plan.")
        for layer_name in descriptor.shared_by:
            if layer_name in descriptors:
                raise ValueError(f"Cache layer {layer_name!r} has multiple independent physical descriptors.")
            descriptors[layer_name] = descriptor

    caches: dict[str, Any] = {}
    for group in attention_groups:
        backend = group.backend
        for layer_name in group.layer_names:
            if layer_name in runner_only_layers:
                continue

            spec = layer_specs[layer_name]
            descriptor = descriptors[layer_name]
            raw_tensor = raw_tensors[layer_name]
            if not isinstance(raw_tensor, torch.Tensor):
                raise TypeError(
                    "Independent cache layers require one raw backing tensor, "
                    f"got {type(raw_tensor).__name__} for {layer_name}."
                )
            if raw_tensor.numel() != descriptor.size:
                raise ValueError(
                    "Raw cache size does not match its independent descriptor: "
                    f"layer={layer_name}, raw={raw_tensor.numel()}, "
                    f"descriptor={descriptor.size}."
                )
            if (
                descriptor.block_stride <= 0
                or descriptor.offset < 0
                or descriptor.offset + spec.page_size_bytes > descriptor.block_stride
            ):
                raise ValueError(
                    "Cache page is outside its independent physical block: "
                    f"layer={layer_name}, offset={descriptor.offset}, "
                    f"page_size={spec.page_size_bytes}, "
                    f"block_stride={descriptor.block_stride}."
                )

            if isinstance(spec, MambaSpec):
                state_tensors = []
                payload_offset = 0
                for shape, dtype in zip(spec.shapes, spec.dtypes):
                    state_tensors.append(
                        _reshape_block_strided_tensor(
                            raw_tensor,
                            (kv_cache_config.num_blocks, *shape),
                            dtype,
                            descriptor.block_stride,
                            descriptor.offset,
                            payload_offset,
                        )
                    )
                    payload_offset += math.prod(shape) * get_dtype_size(dtype)
                if payload_offset > spec.page_size_bytes:
                    raise ValueError(
                        "Mamba states exceed their independent physical page: "
                        f"layer={layer_name}, payload={payload_offset}, "
                        f"page_size={spec.page_size_bytes}."
                    )
                caches[layer_name] = state_tensors
                continue

            if not isinstance(spec, AttentionSpec):
                raise TypeError(f"Unsupported independent cache spec for {layer_name}: {type(spec).__name__}.")

            storage_block_size = getattr(spec, "storage_block_size", spec.block_size)
            if (
                isinstance(spec, AscendMLAAttentionSpec)
                and spec.indexes_kv_by_block_stride
                and spec.compress_ratio == 1
            ):
                k_dim, v_dim = get_attention_cache_dims(layer_name, spec)
                cache_shape = backend.get_kv_cache_shape(
                    kv_cache_config.num_blocks,
                    storage_block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                )
                num_blocks, block_size, num_kv_heads, _ = cache_shape
                k_shape = (num_blocks, block_size, num_kv_heads, k_dim)
                v_shape = (num_blocks, block_size, num_kv_heads, v_dim)
                k_payload_size = math.prod(k_shape[1:]) * get_dtype_size(spec.dtype)
                v_payload_size = math.prod(v_shape[1:]) * get_dtype_size(spec.dtype)
                if k_payload_size + v_payload_size > spec.page_size_bytes:
                    raise ValueError(f"Split MLA payload exceeds its independent physical page for {layer_name}.")
                caches[layer_name] = (
                    _reshape_block_strided_tensor(
                        raw_tensor,
                        k_shape,
                        spec.dtype,
                        descriptor.block_stride,
                        descriptor.offset,
                    ),
                    _reshape_block_strided_tensor(
                        raw_tensor,
                        v_shape,
                        spec.dtype,
                        descriptor.block_stride,
                        descriptor.offset,
                        k_payload_size,
                    ),
                )
                continue

            if spec.compress_ratio > 1 and spec.dtype == torch.uint8:
                indexer_head_size = spec.head_size - get_dtype_size(torch.float16)
                cache_shape = backend.get_kv_cache_shape(
                    kv_cache_config.num_blocks,
                    storage_block_size,
                    spec.num_kv_heads,
                    indexer_head_size,
                )
                scale_shape = backend.get_kv_cache_shape(
                    kv_cache_config.num_blocks,
                    storage_block_size,
                    spec.num_kv_heads,
                    1,
                )
                indexer_payload_size = math.prod(cache_shape[1:]) * get_dtype_size(torch.int8)
                scale_payload_size = math.prod(scale_shape[1:]) * get_dtype_size(torch.float16)
                if indexer_payload_size + scale_payload_size > spec.page_size_bytes:
                    raise ValueError(
                        f"Quantized indexer payload exceeds its independent physical page for {layer_name}."
                    )
                caches[layer_name] = [
                    _reshape_block_strided_tensor(
                        raw_tensor,
                        cache_shape,
                        torch.int8,
                        descriptor.block_stride,
                        descriptor.offset,
                    ),
                    _reshape_block_strided_tensor(
                        raw_tensor,
                        scale_shape,
                        torch.float16,
                        descriptor.block_stride,
                        descriptor.offset,
                        indexer_payload_size,
                    ),
                ]
                continue

            cache_shape = _get_cache_shape(
                backend,
                kv_cache_config.num_blocks,
                storage_block_size,
                spec,
            )
            payload_size = math.prod(cache_shape[1:]) * get_dtype_size(spec.dtype)
            if payload_size > spec.page_size_bytes:
                raise ValueError(
                    "Cache payload exceeds its independent physical page: "
                    f"layer={layer_name}, payload={payload_size}, "
                    f"page_size={spec.page_size_bytes}."
                )
            caches[layer_name] = _reshape_block_strided_tensor(
                raw_tensor,
                cache_shape,
                spec.dtype,
                descriptor.block_stride,
                descriptor.offset,
            )

    return caches
