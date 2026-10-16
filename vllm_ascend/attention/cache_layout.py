# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from vllm.utils.torch_utils import get_dtype_size

if TYPE_CHECKING:
    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec


LAYERWISE_CACHE_LAYOUT_SCHEMA_VERSION = 1
LAYERWISE_PREFILL_GET_GROUP_SIZE = 2
LAYERWISE_PREFILL_WORKSPACE_BANK_COUNT = 2
LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT = (
    LAYERWISE_PREFILL_GET_GROUP_SIZE
    * LAYERWISE_PREFILL_WORKSPACE_BANK_COUNT
)
DEFAULT_CACHE_ALIGNMENT_BYTES = 64


class CacheComponentRole(str, Enum):
    """Semantic role of one physical tensor in an attention KV cache."""

    MAIN_KV = "main_kv"
    MAIN_K_ROPE = "main_k_rope"
    INDEXER_K = "indexer_k"
    INDEXER_SCALE = "indexer_scale"


class MemoryKind(str, Enum):
    DEVICE = "device"
    HOST_DEVICE_VISIBLE = "host_device_visible"


@dataclass(frozen=True)
class CacheComponentDescriptor:
    component_role: CacheComponentRole
    tensor_index: int
    shape_per_block: tuple[int, ...]
    dtype: torch.dtype
    page_bytes: int
    page_stride_bytes: int
    alignment_bytes: int
    prefill_memory_kind: MemoryKind
    decode_memory_kind: MemoryKind

    def __post_init__(self) -> None:
        if self.tensor_index < 0:
            raise ValueError("tensor_index must be non-negative")
        if not self.shape_per_block or any(dim <= 0 for dim in self.shape_per_block):
            raise ValueError(f"Invalid per-block shape: {self.shape_per_block}")
        if self.page_bytes <= 0 or self.page_stride_bytes < self.page_bytes:
            raise ValueError(
                "page_stride_bytes must be greater than or equal to a "
                "positive page_bytes"
            )
        if self.alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be positive")


@dataclass(frozen=True)
class LayerCacheLayoutDescriptor:
    schema_version: int
    layer_name: str
    layer_id: int
    group_id: int
    cache_family_id: str
    layout_id: str
    compatibility_key: str
    is_mtp_layer: bool
    components: tuple[CacheComponentDescriptor, ...]

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.layer_id < 0 or self.group_id < 0:
            raise ValueError("layer_id and group_id must be non-negative")
        if not self.layer_name or not self.cache_family_id or not self.layout_id:
            raise ValueError("layer_name, cache_family_id and layout_id are required")
        roles = [component.component_role for component in self.components]
        indices = [component.tensor_index for component in self.components]
        if len(roles) != len(set(roles)):
            raise ValueError(f"Duplicate cache component role in {self.layer_name}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate tensor index in {self.layer_name}")

    @property
    def main_components(self) -> tuple[CacheComponentDescriptor, ...]:
        return tuple(
            component
            for component in self.components
            if component.component_role
            in (CacheComponentRole.MAIN_KV, CacheComponentRole.MAIN_K_ROPE)
        )

    @property
    def indexer_components(self) -> tuple[CacheComponentDescriptor, ...]:
        return tuple(
            component
            for component in self.components
            if component.component_role
            in (CacheComponentRole.INDEXER_K, CacheComponentRole.INDEXER_SCALE)
        )


@dataclass(frozen=True)
class LayerCacheViews:
    by_role: Mapping[CacheComponentRole, torch.Tensor]

    def __getitem__(self, role: CacheComponentRole) -> torch.Tensor:
        return self.by_role[role]


def _component(
    role: CacheComponentRole,
    tensor_index: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    *,
    alignment_bytes: int,
) -> CacheComponentDescriptor:
    shape = (block_size, num_kv_heads, head_dim)
    page_bytes = block_size * num_kv_heads * head_dim * get_dtype_size(dtype)
    decode_memory_kind = (
        MemoryKind.HOST_DEVICE_VISIBLE
        if role in (CacheComponentRole.MAIN_KV, CacheComponentRole.MAIN_K_ROPE)
        else MemoryKind.DEVICE
    )
    return CacheComponentDescriptor(
        component_role=role,
        tensor_index=tensor_index,
        shape_per_block=shape,
        dtype=dtype,
        page_bytes=page_bytes,
        page_stride_bytes=page_bytes,
        alignment_bytes=alignment_bytes,
        prefill_memory_kind=MemoryKind.DEVICE,
        decode_memory_kind=decode_memory_kind,
    )


def build_mla_layer_cache_descriptor(
    *,
    layer_name: str,
    layer_id: int,
    group_id: int,
    cache_family_id: str,
    kv_cache_spec: AscendMLAAttentionSpec,
    is_mtp_layer: bool = False,
    alignment_bytes: int = DEFAULT_CACHE_ALIGNMENT_BYTES,
) -> LayerCacheLayoutDescriptor:
    """Build the physical cache tuple contract used by Ascend SFA.

    The tuple interpretation lives next to the SFA backend rather than in a
    connector or pool.  A3 non-C8 is ``(latent, rope, indexer_k)``; A3 C8 adds
    ``indexer_scale``; A5 C8 merges the Main cache into one tensor.
    """

    sparse_dims = kv_cache_spec.sparse_head_dim
    if sparse_dims is None or len(sparse_dims) != 3:
        raise ValueError("SFA offload requires a three-part sparse_head_dim")
    kv_lora_rank, qk_rope_head_dim, index_head_dim = sparse_dims
    if min(kv_lora_rank, index_head_dim) <= 0 or qk_rope_head_dim < 0:
        raise ValueError(f"Invalid sparse_head_dim: {sparse_dims}")

    components: list[CacheComponentDescriptor] = []
    if kv_cache_spec.cache_sparse_c8 and qk_rope_head_dim == 0:
        components.append(
            _component(
                CacheComponentRole.MAIN_KV,
                0,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_lora_rank,
                kv_cache_spec.c8_k_cache_dtype,
                alignment_bytes=alignment_bytes,
            )
        )
        components.append(
            _component(
                CacheComponentRole.INDEXER_K,
                1,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                index_head_dim,
                kv_cache_spec.c8_k_cache_dtype,
                alignment_bytes=alignment_bytes,
            )
        )
        components.append(
            _component(
                CacheComponentRole.INDEXER_SCALE,
                2,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                1,
                kv_cache_spec.c8_k_scale_cache_dtype,
                alignment_bytes=alignment_bytes,
            )
        )
    else:
        components.append(
            _component(
                CacheComponentRole.MAIN_KV,
                0,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_lora_rank,
                kv_cache_spec.dtype,
                alignment_bytes=alignment_bytes,
            )
        )
        if qk_rope_head_dim <= 0:
            raise ValueError("A3 SFA layout requires a separate positive RoPE dimension")
        components.append(
            _component(
                CacheComponentRole.MAIN_K_ROPE,
                1,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                qk_rope_head_dim,
                kv_cache_spec.dtype,
                alignment_bytes=alignment_bytes,
            )
        )
        components.append(
            _component(
                CacheComponentRole.INDEXER_K,
                2,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                index_head_dim,
                kv_cache_spec.c8_k_cache_dtype
                if kv_cache_spec.cache_sparse_c8
                else kv_cache_spec.dtype,
                alignment_bytes=alignment_bytes,
            )
        )
        if kv_cache_spec.cache_sparse_c8:
            components.append(
                _component(
                    CacheComponentRole.INDEXER_SCALE,
                    3,
                    kv_cache_spec.block_size,
                    kv_cache_spec.num_kv_heads,
                    1,
                    kv_cache_spec.c8_k_scale_cache_dtype,
                    alignment_bytes=alignment_bytes,
                )
            )

    layout_payload = [
        {
            "role": component.component_role.value,
            "index": component.tensor_index,
            "shape": component.shape_per_block,
            "dtype": str(component.dtype),
            "stride": component.page_stride_bytes,
            "alignment": component.alignment_bytes,
            "prefill": component.prefill_memory_kind.value,
            "decode": component.decode_memory_kind.value,
        }
        for component in components
    ]
    encoded = json.dumps(layout_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    layout_id = f"ascend-sfa-v{LAYERWISE_CACHE_LAYOUT_SCHEMA_VERSION}-{digest}"
    return LayerCacheLayoutDescriptor(
        schema_version=LAYERWISE_CACHE_LAYOUT_SCHEMA_VERSION,
        layer_name=layer_name,
        layer_id=layer_id,
        group_id=group_id,
        cache_family_id=cache_family_id,
        layout_id=layout_id,
        compatibility_key=layout_id,
        is_mtp_layer=is_mtp_layer,
        components=tuple(components),
    )


def bind_layer_cache_views(
    descriptor: LayerCacheLayoutDescriptor,
    raw_cache_tuple: Sequence[torch.Tensor],
) -> LayerCacheViews:
    views: dict[CacheComponentRole, torch.Tensor] = {}
    for component in descriptor.components:
        if component.tensor_index >= len(raw_cache_tuple):
            raise ValueError(
                f"{descriptor.layer_name} is missing cache tensor index "
                f"{component.tensor_index}"
            )
        tensor = raw_cache_tuple[component.tensor_index]
        expected_tail = component.shape_per_block
        if tensor.dtype != component.dtype:
            raise TypeError(
                f"{descriptor.layer_name}/{component.component_role.value} "
                f"dtype is {tensor.dtype}, expected {component.dtype}"
            )
        if tensor.ndim != len(expected_tail) + 1 or tuple(tensor.shape[1:]) != expected_tail:
            raise ValueError(
                f"{descriptor.layer_name}/{component.component_role.value} shape "
                f"is {tuple(tensor.shape)}, expected (num_blocks, {expected_tail})"
            )
        if tensor.stride(0) * tensor.element_size() != component.page_stride_bytes:
            raise ValueError(
                f"{descriptor.layer_name}/{component.component_role.value} has "
                "an incompatible page stride"
            )
        views[component.component_role] = tensor
    return LayerCacheViews(views)


def resolve_dsa_main_cache_tensors(
    descriptor: LayerCacheLayoutDescriptor,
    raw_cache_tuple: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve DsaServe Main-cache inputs by semantic layout role.

    A3 exposes separate Main KV and RoPE tensors. A5 sparse-C8 stores both in
    the merged Main KV tensor, while tuple index 1 is its Indexer K cache.
    """

    views = bind_layer_cache_views(descriptor, raw_cache_tuple)
    main_kv = views[CacheComponentRole.MAIN_KV]
    main_k_rope = views.by_role.get(CacheComponentRole.MAIN_K_ROPE, main_kv)
    return main_kv, main_k_rope


def stable_model_fingerprint(model_identity: Mapping[str, Any]) -> str:
    """Return a process-independent fingerprint for Store key isolation."""

    encoded = json.dumps(
        model_identity,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()
