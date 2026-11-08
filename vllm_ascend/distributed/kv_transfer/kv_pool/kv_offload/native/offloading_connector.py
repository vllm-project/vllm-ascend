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
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    derive_canonical_mappings,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    CanonicalPageMapping,
)

from vllm_ascend.utils import get_kv_cache_tensor_layers


def _make_int8_block_view(
    tensor: torch.Tensor,
    num_blocks: int,
    physical_blocks_per_manager_block: int,
    bytes_per_physical_block: int,
) -> torch.Tensor:
    """Build a zero-copy ``[num_blocks, page_bytes]`` cache view."""
    required_blocks = num_blocks * physical_blocks_per_manager_block
    if tensor.ndim < 1 or tensor.shape[0] < required_blocks:
        raise ValueError(
            "KV cache tensor has too few physical blocks: "
            f"shape={tuple(tensor.shape)}, num_blocks={num_blocks}, "
            "physical_blocks_per_manager_block="
            f"{physical_blocks_per_manager_block}"
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
    if physical_stride_bytes < bytes_per_physical_block:
        raise ValueError(
            "Ascend KV cache blocks overlap in storage: "
            f"stride_bytes={physical_stride_bytes}, "
            f"physical_block_bytes={bytes_per_physical_block}"
        )
    if physical_blocks_per_manager_block > 1 and physical_stride_bytes != bytes_per_physical_block:
        raise ValueError(
            "Cannot coalesce a non-contiguous Ascend KV cache layout: "
            f"stride_bytes={physical_stride_bytes}, "
            f"physical_block_bytes={bytes_per_physical_block}"
        )

    page_size_bytes = bytes_per_physical_block * physical_blocks_per_manager_block
    manager_stride_bytes = physical_stride_bytes * physical_blocks_per_manager_block
    byte_offset = tensor.storage_offset() * element_size
    raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(tensor.untyped_storage())
    return torch.as_strided(
        raw,
        (num_blocks, page_size_bytes),
        (manager_stride_bytes, 1),
        byte_offset,
    )


def _coalesce_interleaved_slots(
    parts: tuple[tuple[torch.Tensor, int], ...],
    num_blocks: int,
    factor: int,
) -> tuple[tuple[torch.Tensor, int], ...] | None:
    """Use one byte range for K/V components that tile each kernel slot."""
    first, first_bytes = parts[0]
    storage_ptr = first.untyped_storage().data_ptr()
    stride_bytes = first.stride(0) * first.element_size()
    slot_bytes = sum(part_bytes for _, part_bytes in parts)
    if slot_bytes > stride_bytes:
        return None

    offset_bytes = first.storage_offset() * first.element_size()
    next_offset = offset_bytes
    for part, part_bytes in parts:
        element_size = part.element_size()
        if (
            part.device != first.device
            or part.untyped_storage().data_ptr() != storage_ptr
            or part.storage_offset() * element_size != next_offset
            or part.stride(0) * element_size != stride_bytes
        ):
            return None
        next_offset += part_bytes

    # Validate each component's shape, bounds and inner contiguity before
    # exposing the combined range. Byte offsets preserve mixed-dtype views.
    physical_blocks = num_blocks * factor
    first_view = _make_int8_block_view(first, physical_blocks, 1, first_bytes)
    for part, part_bytes in parts[1:]:
        _make_int8_block_view(part, physical_blocks, 1, part_bytes)
    if stride_bytes == slot_bytes:
        page_bytes = slot_bytes * factor
        page = torch.as_strided(first_view, (num_blocks, page_bytes), (page_bytes, 1))
        return ((page, page_bytes),)

    # A padded slot still has a contiguous K/V payload. Keep the padding out
    # of the copy and advance by a full manager page between rows.
    slots = torch.as_strided(first_view, (physical_blocks, slot_bytes), (stride_bytes, 1))
    return tuple((slots[slot::factor], slot_bytes) for slot in range(factor))


def _canonicalize_split_cache(
    cache_parts: Sequence[torch.Tensor],
    num_blocks: int,
    unpadded_page_size_bytes: int,
    *,
    coalesce_interleaved: bool = False,
) -> tuple[tuple[torch.Tensor, int], ...]:
    """Canonicalize separate K/V, scale, or recurrent-state tensors."""
    parts = tuple(cache_parts)
    if not parts or any(not isinstance(part, torch.Tensor) for part in parts):
        raise TypeError("An Ascend KV cache must contain one or more tensors")

    if coalesce_interleaved and any(part.ndim == 0 for part in parts):
        raise ValueError("Ascend KV cache components must have a block dimension")

    part_block_bytes = tuple(math.prod(part.shape[1:]) * part.element_size() for part in parts)
    # Strided aliases can span more bytes than their payload element count.
    part_block_spans = (
        tuple(
            (1 + sum((size - 1) * stride for size, stride in zip(part.shape[1:], part.stride()[1:])))
            * part.element_size()
            for part in parts
        )
        if coalesce_interleaved
        else part_block_bytes
    )
    if any(size <= 0 for size in part_block_bytes):
        raise ValueError("Ascend KV cache components must be non-empty")

    # Some backends expose component views and an overlapping full-page view.
    # Verify byte coverage before dropping component views. On the MRV2
    # path, also verify that this coverage repeats for every physical block.
    selected: tuple[tuple[torch.Tensor, int], ...] | None = None
    for part, part_bytes in sorted(zip(parts, part_block_bytes), key=lambda item: item[1], reverse=True):
        if unpadded_page_size_bytes % part_bytes:
            continue
        factor = unpadded_page_size_bytes // part_bytes
        physical_blocks = num_blocks * factor
        if part.shape[0] < physical_blocks:
            continue
        if coalesce_interleaved:
            stride_bytes = part.stride(0) * part.element_size()
            if any(
                other.shape[0] < physical_blocks
                or (physical_blocks > 1 and other.stride(0) * other.element_size() != stride_bytes)
                for other in parts
            ):
                continue

        candidate_start = part.data_ptr()
        candidate_end = candidate_start + part_bytes
        candidate_storage_ptr = part.untyped_storage().data_ptr()
        contains_all_parts = all(
            other.untyped_storage().data_ptr() == candidate_storage_ptr
            and candidate_start <= other.data_ptr()
            and other.data_ptr() + other_span <= candidate_end
            for other, other_span in zip(parts, part_block_spans)
        )
        if contains_all_parts:
            selected = ((part, part_bytes),)
            break

    if selected is None:
        bytes_per_physical_block = sum(part_block_bytes)
        if unpadded_page_size_bytes % bytes_per_physical_block:
            raise ValueError(
                "Ascend KV cache components do not cover one logical page: "
                f"component_bytes={part_block_bytes}, "
                f"page_size_bytes={unpadded_page_size_bytes}"
            )
        selected = tuple(zip(parts, part_block_bytes))

    bytes_per_physical_block = sum(size for _, size in selected)
    factor = unpadded_page_size_bytes // bytes_per_physical_block
    if coalesce_interleaved and factor > 1 and len(selected) > 1:
        combined = _coalesce_interleaved_slots(selected, num_blocks, factor)
        if combined is not None:
            return combined

    views: list[tuple[torch.Tensor, int]] = []
    for part, part_bytes in selected:
        if factor > 1 and part.stride(0) * part.element_size() > part_bytes:
            # Interleaved K/V or padded slots cannot be coalesced into one
            # contiguous payload. Each view selects the same kernel slot in
            # every manager block, preserving gaps without copying them.
            for slot in range(factor):
                view = _make_int8_block_view(part[slot::factor], num_blocks, 1, part_bytes)
                views.append((view, part_bytes))
        else:
            view = _make_int8_block_view(part, num_blocks, factor, part_bytes)
            views.append((view, part_bytes * factor))
    return tuple(views)


def _deduplicate_overlapping_views(kv_caches: CanonicalKVCaches) -> CanonicalKVCaches:
    """Share CPU slots for overlapping pages of hybrid cache groups.

    Attention and Mamba layers can overlay the same pool with different
    component boundaries. Exact-view deduplication alone counts those bytes
    twice. Partition each common block stride at component boundaries so
    every reference still copies exactly its own payload.
    """
    pools: dict[tuple[int, int], list[int]] = {}
    for idx, cache in enumerate(kv_caches.tensors):
        tensor = cache.tensor
        pools.setdefault((tensor.untyped_storage().data_ptr(), tensor.stride(0)), []).append(idx)

    tensors: list[CanonicalKVCacheTensor] = []
    replacements: dict[int, list[tuple[int, int, int]]] = {}
    for indices in pools.values():
        boundaries = sorted(
            {
                offset
                for idx in indices
                for offset in (
                    kv_caches.tensors[idx].tensor.storage_offset(),
                    kv_caches.tensors[idx].tensor.storage_offset() + kv_caches.tensors[idx].page_size_bytes,
                )
            }
        )
        for start, end in zip(boundaries, boundaries[1:]):
            owners = [
                idx
                for idx in indices
                if kv_caches.tensors[idx].tensor.storage_offset() <= start
                and end <= kv_caches.tensors[idx].tensor.storage_offset() + kv_caches.tensors[idx].page_size_bytes
            ]
            if not owners:
                continue
            source = kv_caches.tensors[owners[0]].tensor
            tensor_idx = len(tensors)
            tensors.append(
                CanonicalKVCacheTensor(
                    tensor=torch.as_strided(source, (source.shape[0], end - start), source.stride(), start),
                    page_size_bytes=end - start,
                )
            )
            for idx in owners:
                replacements.setdefault(idx, []).append(
                    (tensor_idx, start - kv_caches.tensors[idx].tensor.storage_offset(), end - start)
                )

    group_refs: list[list[CanonicalKVCacheRef]] = []
    for refs in kv_caches.group_data_refs:
        new_refs: list[CanonicalKVCacheRef] = []
        for ref in refs:
            pieces = replacements[ref.tensor_idx]
            for tensor_idx, offset, size in pieces:
                copy_size = min(size, ref.page_size_bytes - offset)
                if copy_size > 0:
                    new_refs.append(
                        CanonicalKVCacheRef(
                            tensor_idx=tensor_idx,
                            page_size_bytes=copy_size,
                            mapping=ref.mapping if len(pieces) == 1 else None,
                        )
                    )
        group_refs.append(new_refs)
    return CanonicalKVCaches(tensors=tensors, group_data_refs=group_refs)


class AscendOffloadingConnectorWorker(OffloadingConnectorWorker):
    """Offloading worker boundary supporting Ascend's split KV layouts."""

    def register_kv_caches(
        self,
        kv_caches: dict[
            str,
            torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        ],
    ) -> None:
        kv_cache_config = self.kv_cache_config
        # Only the groups the scheduler actually offloads, i.e. the same
        # selection as vLLM's native offloading worker. Hybrid models expose
        # extra non-prefix-cacheable groups (e.g. DeepSeek-V4.1 Engram and
        # sparse attention); registering those here would make the group count
        # diverge from GPULoadStoreSpec.group_sizes and misalign block ids.
        selected_groups = tuple(
            kv_cache_config.kv_cache_groups[group_id] for group_id in get_offloading_group_ids(kv_cache_config)
        )

        requires_layout_adaptation = False
        for group in selected_groups:
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

        if not requires_layout_adaptation:
            super().register_kv_caches(kv_caches)  # type: ignore[arg-type]
            return

        num_blocks = kv_cache_config.num_blocks
        coalesce_interleaved = self.vllm_config.use_v2_model_runner
        mappings = derive_canonical_mappings(
            self.vllm_config,
            kv_cache_config,
            kv_caches,  # type: ignore[arg-type]
        )
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
            mapping: CanonicalPageMapping | None = None,
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
            if mapping is not None:
                assert mapping.local_page_size_bytes == copy_size_bytes
            refs_by_layer.setdefault(layer_name, []).append(
                CanonicalKVCacheRef(
                    tensor_idx=tensor_idx,
                    page_size_bytes=copy_size_bytes,
                    mapping=mapping,
                )
            )

        for group in selected_groups:
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
                        raw = torch.empty(0, dtype=torch.int8, device=layer_cache.device).set_(
                            layer_cache.untyped_storage()
                        )
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
                            mappings.get(layer_name),
                        )
                    elif isinstance(layer_cache, (tuple, list)):
                        views = _canonicalize_split_cache(
                            layer_cache,
                            num_blocks,
                            layer_spec.unpadded_page_size_bytes,
                            coalesce_interleaved=coalesce_interleaved and type(layer_spec) is FullAttentionSpec,
                        )
                        # A page mapping describes the complete local page and
                        # is valid only when one canonical view covers it. For
                        # genuinely split K/V refs, leave mapping unset rather
                        # than applying incorrect offsets to every component.
                        mapping = mappings.get(layer_name) if len(views) == 1 else None
                        for view, copy_size_bytes in views:
                            add_view(
                                layer_name,
                                view,
                                copy_size_bytes,
                                mapping,
                            )
                    else:
                        raise TypeError(f"Unsupported KV cache type for {layer_name}: {type(layer_cache).__name__}")
                elif isinstance(layer_spec, MambaSpec):
                    if not isinstance(layer_cache, list) or not layer_cache:
                        raise TypeError(f"Mamba KV cache for {layer_name} must be a non-empty list")
                    unpadded_page_size_bytes = sum(
                        math.prod(state.shape[1:]) * state.element_size() for state in layer_cache
                    )
                    expected_page_size_bytes = replace(layer_spec, page_size_padded=None).page_size_bytes
                    if unpadded_page_size_bytes != expected_page_size_bytes:
                        raise ValueError(
                            f"Mamba KV cache page size mismatch for {layer_name}: "
                            f"tensors={unpadded_page_size_bytes}, "
                            f"spec={expected_page_size_bytes}"
                        )
                    for view, copy_size_bytes in _canonicalize_split_cache(
                        layer_cache,
                        num_blocks,
                        unpadded_page_size_bytes,
                    ):
                        add_view(layer_name, view, copy_size_bytes)
                else:
                    raise NotImplementedError(f"Unsupported KV cache spec: {type(layer_spec).__name__}")

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for group in selected_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in group.layer_names:
                group_refs.extend(refs_by_layer[layer_name])
            group_data_refs.append(group_refs)

        if not canonical_tensors:
            raise ValueError("No KV cache tensors were registered for native offloading")
        self._init_worker(
            _deduplicate_overlapping_views(
                CanonicalKVCaches(
                    tensors=canonical_tensors,
                    group_data_refs=group_data_refs,
                )
            )
        )


class AscendOffloadingConnector(OffloadingConnector):
    """vLLM OffloadingConnector with an Ascend layout adapter."""

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
            # An opt-in synchronous load completes in pre_forward. Defer the
            # MRV2 state pre-copy so it reads the checkpoint after that load.
            self.requires_mamba_state_copy_after_load = (
                vllm_config.use_v2_model_runner
                and int(connector_worker.spec.extra_config.get("sync_load_budget_bytes", 0)) > 0
                and kv_cache_config.has_mamba_layers
            )
            self.connector_worker = AscendOffloadingConnectorWorker(
                connector_worker.spec,
                vllm_config,
                kv_cache_config,
            )
