# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend layout adaptation for vLLM's native offloading connector."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)

from vllm_ascend.utils import get_kv_cache_tensor_layers


def _make_int8_block_view(
    tensor: torch.Tensor,
    num_blocks: int,
    logical_blocks_per_manager_block: int,
    bytes_per_logical_block: int,
) -> torch.Tensor:
    """Build a zero-copy ``[num_blocks, page_bytes]`` view of a cache part."""
    if tensor.ndim < 1 or tensor.shape[0] < num_blocks * logical_blocks_per_manager_block:
        raise ValueError(
            "KV cache tensor has too few physical blocks: "
            f"shape={tuple(tensor.shape)}, num_blocks={num_blocks}, "
            f"logical_blocks_per_manager_block={logical_blocks_per_manager_block}"
        )

    element_size = tensor.element_size()
    physical_stride_bytes = tensor.stride(0) * element_size
    expected_stride = 1
    for size, stride in zip(reversed(tensor.shape[1:]), reversed(tensor.stride()[1:])):
        if size > 1 and stride != expected_stride:
            raise ValueError(
                "Cannot offload an Ascend KV cache whose block payload is "
                f"non-contiguous: shape={tuple(tensor.shape)}, "
                f"stride={tensor.stride()}"
            )
        expected_stride *= size
    if physical_stride_bytes < bytes_per_logical_block:
        raise ValueError(
            "Ascend KV cache blocks overlap in storage: "
            f"stride_bytes={physical_stride_bytes}, "
            f"logical_block_bytes={bytes_per_logical_block}"
        )
    if logical_blocks_per_manager_block > 1 and physical_stride_bytes != bytes_per_logical_block:
        raise ValueError(
            "Cannot coalesce a non-contiguous Ascend KV cache layout: "
            f"stride_bytes={physical_stride_bytes}, "
            f"logical_block_bytes={bytes_per_logical_block}"
        )

    page_size_bytes = bytes_per_logical_block * logical_blocks_per_manager_block
    manager_stride_bytes = physical_stride_bytes * logical_blocks_per_manager_block
    byte_offset = tensor.storage_offset() * element_size
    raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(tensor.untyped_storage())
    return torch.as_strided(
        raw,
        (num_blocks, page_size_bytes),
        (manager_stride_bytes, 1),
        byte_offset,
    )


def _canonicalize_split_attention_cache(
    cache_parts: Sequence[torch.Tensor],
    num_blocks: int,
    unpadded_page_size_bytes: int,
) -> tuple[tuple[torch.Tensor, int], ...]:
    """Canonicalize Ascend's separate K/V (and optional scale) tensors.

    Ascend attention keeps cache components as separate tensors while vLLM's
    native offloading worker expects a canonical block-level representation.
    The returned views alias the original NPU storage and therefore add no
    allocation or copy on the model execution path.
    """
    parts = tuple(cache_parts)
    if not parts or any(not isinstance(part, torch.Tensor) for part in parts):
        raise TypeError("An Ascend attention KV cache must contain one or more tensors")

    logical_block_bytes = tuple(math.prod(part.shape[1:]) * part.element_size() for part in parts)
    if any(size <= 0 for size in logical_block_bytes):
        raise ValueError("Ascend attention KV cache components must be non-empty")

    # Some Ascend layouts expose both component views and a full overlapping
    # view. Prefer the largest single view only when its first logical block
    # actually contains every component's first block. Shape alone is not
    # sufficient: a runner can allocate more physical blocks than the cache
    # manager uses, which would otherwise make a long K tensor look like a
    # complete K/V view and silently omit V from offloading.
    selected: tuple[tuple[torch.Tensor, int], ...] | None = None
    for part, part_bytes in sorted(zip(parts, logical_block_bytes), key=lambda item: item[1], reverse=True):
        if unpadded_page_size_bytes % part_bytes:
            continue
        factor = unpadded_page_size_bytes // part_bytes
        candidate_start = part.data_ptr()
        candidate_end = candidate_start + part_bytes
        candidate_storage_ptr = part.untyped_storage().data_ptr()
        contains_all_parts = all(
            other.untyped_storage().data_ptr() == candidate_storage_ptr
            and candidate_start <= other.data_ptr()
            and other.data_ptr() + other_bytes <= candidate_end
            for other, other_bytes in zip(parts, logical_block_bytes)
        )
        if contains_all_parts and part.shape[0] >= num_blocks * factor:
            selected = ((part, part_bytes),)
            break

    if selected is None:
        bytes_per_physical_block = sum(logical_block_bytes)
        if unpadded_page_size_bytes % bytes_per_physical_block:
            raise ValueError(
                "Ascend KV cache components do not cover one logical page: "
                f"component_bytes={logical_block_bytes}, "
                f"page_size_bytes={unpadded_page_size_bytes}"
            )
        selected = tuple(zip(parts, logical_block_bytes))

    bytes_per_physical_block = sum(size for _, size in selected)
    factor = unpadded_page_size_bytes // bytes_per_physical_block
    views: list[tuple[torch.Tensor, int]] = []
    for part, part_bytes in selected:
        page_bytes = part_bytes * factor
        views.append(
            (
                _make_int8_block_view(
                    part,
                    num_blocks,
                    factor,
                    part_bytes,
                ),
                page_bytes,
            )
        )
    return tuple(views)


class AscendOffloadingConnectorWorker(OffloadingConnectorWorker):
    """Offloading worker boundary supporting Ascend's separate K/V layout."""

    def register_kv_caches(
        self,
        kv_caches: dict[
            str,
            torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        ],
    ) -> None:
        kv_cache_config = self.kv_cache_config

        requires_layout_adaptation = False
        for group in kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            per_layer_specs = group_spec.kv_cache_specs if isinstance(group_spec, UniformTypeKVCacheSpecs) else {}
            for layer_name in group.layer_names:
                layer_spec = per_layer_specs.get(layer_name, group_spec)
                layer_cache = kv_caches[layer_name]
                if (isinstance(layer_spec, AttentionSpec) and not isinstance(layer_cache, torch.Tensor)) or (
                    isinstance(layer_spec, MambaSpec) and isinstance(layer_cache, list)
                ):
                    requires_layout_adaptation = True
                    break
            if requires_layout_adaptation:
                break

        # Preserve upstream behavior verbatim for upstream-compatible layouts.
        if not requires_layout_adaptation:
            super().register_kv_caches(kv_caches)  # type: ignore[arg-type]
            return

        num_blocks = kv_cache_config.num_blocks
        layer_is_packed = {
            layer_name: bool(kv_tensor.block_stride)
            for kv_tensor in kv_cache_config.kv_cache_tensors
            for layer_name in get_kv_cache_tensor_layers(kv_tensor)
        }

        canonical_tensors: list[CanonicalKVCacheTensor] = []
        tensor_indices: dict[tuple[int, int, int], int] = {}
        refs_by_layer: dict[str, list[CanonicalKVCacheRef]] = {}

        def add_view(
            layer_name: str,
            tensor: torch.Tensor,
            copy_size_bytes: int,
        ) -> None:
            key = (tensor.data_ptr(), tensor.stride(0), tensor.shape[1])
            tensor_idx = tensor_indices.get(key)
            if tensor_idx is None:
                tensor_idx = len(canonical_tensors)
                tensor_indices[key] = tensor_idx
                canonical_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=tensor,
                        page_size_bytes=tensor.shape[1],
                    )
                )
            refs_by_layer.setdefault(layer_name, []).append(
                CanonicalKVCacheRef(
                    tensor_idx=tensor_idx,
                    page_size_bytes=copy_size_bytes,
                )
            )

        for group in kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            per_layer_specs = group_spec.kv_cache_specs if isinstance(group_spec, UniformTypeKVCacheSpecs) else {}
            for layer_name in group.layer_names:
                if layer_name in refs_by_layer:
                    continue
                layer_spec = per_layer_specs.get(layer_name, group_spec)
                layer_cache = kv_caches[layer_name]

                if isinstance(layer_spec, AttentionSpec):
                    if isinstance(layer_cache, torch.Tensor):
                        page_size_bytes = layer_spec.page_size_bytes
                        element_size = layer_cache.element_size()
                        block_stride_bytes = (
                            layer_cache.stride(0) * element_size
                            if layer_is_packed.get(layer_name, False)
                            else page_size_bytes
                        )
                        raw = torch.empty(
                            0,
                            dtype=torch.int8,
                            device=layer_cache.device,
                        ).set_(layer_cache.untyped_storage())
                        view = torch.as_strided(
                            raw,
                            (num_blocks, page_size_bytes),
                            (block_stride_bytes, 1),
                            layer_cache.storage_offset() * element_size,
                        )
                        add_view(
                            layer_name,
                            view,
                            layer_spec.unpadded_page_size_bytes,
                        )
                    elif isinstance(layer_cache, (tuple, list)):
                        for view, copy_size_bytes in _canonicalize_split_attention_cache(
                            layer_cache,
                            num_blocks,
                            layer_spec.unpadded_page_size_bytes,
                        ):
                            add_view(layer_name, view, copy_size_bytes)
                    else:
                        raise TypeError(f"Unsupported KV cache type for {layer_name}: {type(layer_cache).__name__}")
                elif isinstance(layer_spec, MambaSpec):
                    if not isinstance(layer_cache, list) or not layer_cache:
                        raise TypeError(f"Mamba KV cache for {layer_name} must be a non-empty list")
                    # Ascend lays Mamba states out as consecutive, independent
                    # ``[num_blocks, ...]`` regions and may align the underlying
                    # raw allocation, yielding a non-zero storage offset. The
                    # upstream reconstruction from state[0]'s whole storage
                    # assumes block-interleaved states starting at offset zero.
                    # Register each logical state instead; this is zero-copy and
                    # transfers exactly the unpadded state bytes.
                    unpadded_page_size_bytes = sum(
                        math.prod(state.shape[1:]) * state.element_size() for state in layer_cache
                    )
                    expected_page_size_bytes = replace(
                        layer_spec,
                        page_size_padded=None,
                    ).page_size_bytes
                    if unpadded_page_size_bytes != expected_page_size_bytes:
                        raise ValueError(
                            f"Mamba KV cache page size mismatch for {layer_name}: "
                            f"tensors={unpadded_page_size_bytes}, "
                            f"spec={expected_page_size_bytes}"
                        )
                    for view, copy_size_bytes in _canonicalize_split_attention_cache(
                        layer_cache,
                        num_blocks,
                        unpadded_page_size_bytes,
                    ):
                        add_view(layer_name, view, copy_size_bytes)
                else:
                    raise NotImplementedError(f"Unsupported KV cache spec: {type(layer_spec).__name__}")

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for group in kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in group.layer_names:
                group_refs.extend(refs_by_layer[layer_name])
            group_data_refs.append(group_refs)

        if not canonical_tensors:
            raise ValueError("No KV cache tensors were registered for native offloading")
        self._init_worker(
            CanonicalKVCaches(
                tensors=canonical_tensors,
                group_data_refs=group_data_refs,
            )
        )


class AscendOffloadingConnector(OffloadingConnector):
    """vLLM OffloadingConnector with an Ascend layout adapter."""

    # vLLM is skipped by the Ascend mypy invocation, so inherited attributes
    # are not visible to the type checker. Keep this aligned with upstream;
    # ``super().__init__`` still owns initialization and role handling.
    connector_worker: OffloadingConnectorWorker | None

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        connector_worker = self.connector_worker
        if connector_worker is not None:
            self.connector_worker = AscendOffloadingConnectorWorker(
                connector_worker.spec,
                vllm_config,
                kv_cache_config,
            )
