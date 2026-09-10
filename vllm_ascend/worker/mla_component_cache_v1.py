# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The vllm-ascend contributors
"""Allocation and validation of the V1 MLA component-major cache.

The dense path keeps the original DeepSeek-V3 contract: one exact MLA group,
one backing, and one unpadded component-major page per kernel block.  The K3
hybrid path keeps MLA and Mamba/KDA regions in the standardized HMA backing,
splits each MLA manager page into component-major kernel slots, and leaves
Mamba/KDA state materialization to the legacy runner path.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheLayout,
    KVCacheTensor,
    KVQuantMode,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend import envs
from vllm_ascend.utils import vllm_version_is


@dataclass(frozen=True)
class _MLAGeometry:
    block_size: int
    num_kv_heads: int
    nope_dim: int
    rope_dim: int
    dtype: torch.dtype

    @property
    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def nope_segment_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.nope_dim * self.element_size

    @property
    def rope_segment_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.rope_dim * self.element_size

    @property
    def page_bytes(self) -> int:
        return self.nope_segment_bytes + self.rope_segment_bytes


@dataclass(frozen=True)
class MLAComponentCacheCapability:
    """The component-cache mode selected before the KV-cache plan exists."""

    mode: str
    mla_layer_names: tuple[str, ...]
    manager_block_size: int
    kernel_block_size: int


def _feature_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE)


def _unsupported(reason: str) -> ValueError:
    return ValueError(
        "VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE=1 is enabled, but the model "
        f"does not satisfy the MLA component-cache capability: {reason}"
    )


def _validate_exact_spec(spec: Any, layer_name: str) -> None:
    """Require an unpadded, exact, dense upstream MLA spec."""
    # 这里校验的是MLA layer在hybrid page统一前的logical spec：
    # 必须是没有任何扩展、量化、压缩、padding和sliding window的exact upstream MLAAttentionSpec。
    if type(spec) is not MLAAttentionSpec:
        raise _unsupported(
            f"layer {layer_name} returned {type(spec).__name__}, expected the exact upstream MLAAttentionSpec"
        )

    geometry_fields = (
        "page_size_padded",
        "num_head_slots",
        "state_content_bytes",
        "storage_block_size",
        "alignment",
        "model_version",
    )
    for field_name in geometry_fields:
        if getattr(spec, field_name, None) is not None:
            raise _unsupported(f"layer {layer_name} sets {field_name}")
    if spec.kv_quant_mode != KVQuantMode.NONE:
        raise _unsupported(f"layer {layer_name} uses a quantized KV cache")
    if spec.tokens_per_state != 1:
        raise _unsupported(f"layer {layer_name} has tokens_per_state={spec.tokens_per_state}")
    if spec.sliding_window is not None or spec.attention_chunk_size is not None:
        raise _unsupported(f"layer {layer_name} is not dense full attention")
    if spec.head_size_v != 0:
        raise _unsupported(f"layer {layer_name} has head_size_v={spec.head_size_v}")
    if spec.block_size <= 0 or spec.num_kv_heads <= 0 or spec.head_size <= 0:
        raise _unsupported(f"layer {layer_name} has non-positive cache geometry")


def _layer_geometry(
    layer: MLAAttention,
    spec: MLAAttentionSpec,
    layer_name: str,
) -> _MLAGeometry:
    """Derive the logical N/H/Dk/Dr geometry from an MLA layer and its spec."""
    # 构造NHD的描述对象，并校验spec满足nope+rope
    _validate_exact_spec(spec, layer_name)
    nope_dim = layer.kv_lora_rank
    rope_dim = layer.qk_rope_head_dim
    if nope_dim <= 0 or rope_dim <= 0:
        raise _unsupported(
            f"layer {layer_name} has kv_lora_rank={nope_dim} and qk_rope_head_dim={rope_dim}; MLA-NoPE is not supported"
        )
    if layer.head_size != nope_dim + rope_dim:
        raise _unsupported(f"layer {layer_name} has inconsistent MLA head dimensions")
    if getattr(layer, "num_kv_heads", spec.num_kv_heads) != spec.num_kv_heads:
        raise _unsupported(f"layer {layer_name} has inconsistent num_kv_heads")
    geometry = _MLAGeometry(
        block_size=spec.block_size,
        num_kv_heads=spec.num_kv_heads,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        dtype=spec.dtype,
    )

    expected_page_bytes = geometry.block_size * geometry.num_kv_heads * (nope_dim + rope_dim) * geometry.element_size
    if spec.head_size != nope_dim + rope_dim or spec.page_size_bytes != expected_page_bytes:
        raise _unsupported(f"layer {layer_name} spec has an inconsistent page size")
    return geometry


def _select_kernel_block_size(manager_block_size: int, supported_block_sizes: Sequence[int]) -> int:
    divisors = [size for size in supported_block_sizes if isinstance(size, int) and manager_block_size % size == 0]
    if not divisors:
        raise _unsupported(
            f"manager block size {manager_block_size} cannot be split by any supported "
            f"kernel block size {list(supported_block_sizes)}"
        )
    return max(divisors)


def get_mla_component_cache_capability(
    vllm_config: VllmConfig,
) -> MLAComponentCacheCapability | None:
    """Classify a model as dense-V1 or K3-hybrid-V1 before grouping."""
    if not _feature_enabled():
        return None

    if getattr(vllm_config, "use_v2_model_runner", False):
        raise _unsupported("ModelRunner V2 is outside the V1-only scope")
    if vllm_version_is("0.28.0"):
        raise _unsupported("the standardized KVCacheTensor plan requires vLLM main")
    if vllm_config.kv_transfer_config is not None:
        raise _unsupported("KV transfer/offload is not supported")

    model_config = getattr(vllm_config, "model_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hasattr(hf_text_config, "compress_ratios"):
        raise _unsupported("compressed MLA cache is not supported")

    forward_context = vllm_config.compilation_config.static_forward_context
    if not isinstance(forward_context, Mapping):
        raise _unsupported("the static forward context is not a mapping")

    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    mla_layers: dict[str, MLAAttention] = {}
    has_mamba_cache = False
    for layer_name, layer in layers.items():
        if getattr(layer, "kv_sharing_target_layer_name", None) is not None:
            raise _unsupported(f"layer {layer_name} uses cross-layer KV sharing")
        if isinstance(layer, MLAAttention):
            mla_layers[layer_name] = layer
            continue

        try:
            spec = layer.get_kv_cache_spec(vllm_config)
        except Exception as exc:
            raise _unsupported(f"cannot inspect the KV cache spec of non-MLA layer {layer_name}") from exc
        if spec is None:
            continue
        if isinstance(spec, MambaSpec):
            has_mamba_cache = True
            continue
        if isinstance(spec, EncoderOnlyAttentionSpec):
            # Encoder-only attention is runner-only and has no persistent KV
            # cache allocation, so it neither participates in nor blocks the
            # component-cache capability.
            continue
        raise _unsupported(
            f"layer {layer_name} is {type(layer).__name__} with {type(spec).__name__}; "
            "only MLA and Mamba/KDA cache groups are supported"
        )

    if not mla_layers:
        raise _unsupported("the model has no MLA layers")
    if has_mamba_cache and getattr(vllm_config, "speculative_config", None) is not None:
        raise _unsupported("speculative decoding is outside the first K3 hybrid capability")

    first_geometry: _MLAGeometry | None = None
    first_spec: MLAAttentionSpec | None = None
    selected_kernel_size: int | None = None
    for layer_name, layer in mla_layers.items():
        impl = layer.impl
        if getattr(impl, "fa_quant_layer", False):
            raise _unsupported(f"layer {layer_name} uses FA quant")
        if getattr(impl, "enable_mlapo", False):
            raise _unsupported(f"layer {layer_name} uses MLAPO")
        if getattr(layer, "indexer", None) is not None:
            raise _unsupported(f"layer {layer_name} has a sparse indexer")

        try:
            spec = layer.get_kv_cache_spec(vllm_config)
            backend = layer.get_attn_backend()
            spec = backend.customize_spec(spec)
        except Exception as exc:
            raise _unsupported(f"cannot inspect the MLA cache spec of {layer_name}") from exc
        geometry = _layer_geometry(layer, spec, layer_name)

        supported_layouts = backend.supported_kv_cache_layouts()
        if supported_layouts is None or KVCacheLayout.LBNHC not in supported_layouts:
            raise _unsupported(f"layer {layer_name} backend does not support LBNHC")

        kernel_size = _select_kernel_block_size(
            geometry.block_size,
            backend.get_supported_kernel_block_sizes(),
        )
        if selected_kernel_size is None:
            selected_kernel_size = kernel_size
        elif kernel_size != selected_kernel_size:
            raise _unsupported(f"layer {layer_name} selects a different kernel block size")

        if first_geometry is None:
            first_geometry = geometry
            first_spec = spec
        elif geometry != first_geometry or spec != first_spec:
            raise _unsupported(f"layer {layer_name} has different MLA cache geometry")

    assert first_geometry is not None and first_spec is not None and selected_kernel_size is not None
    if not has_mamba_cache and first_geometry.block_size != selected_kernel_size:
        raise _unsupported(
            f"dense MLA requires manager and kernel block sizes to match, got manager={first_geometry.block_size} "
            f"kernel={selected_kernel_size}"
        )
    return MLAComponentCacheCapability(
        mode="K3_HYBRID_V1" if has_mamba_cache else "DENSE_V1",
        mla_layer_names=tuple(mla_layers),
        manager_block_size=first_geometry.block_size,
        kernel_block_size=selected_kernel_size,
    )


def use_mla_component_cache(vllm_config: VllmConfig) -> bool:
    """Return whether the V1 runner can use the component-major MLA cache.

    判断当前是否使用MLA首轴非连续能力。
    feature开启时先做结构化能力分类：
    1. DeepSeek V3类dense MLA归入DENSE_V1；
    2. K3类“MLA subset + Mamba/KDA subset”归入K3_HYBRID_V1；
    3. 两条路径都使用component-major nope/rope布局；
    4. 不满足能力边界时启动阶段直接失败，避免静默fallback到不兼容路径。
    """
    return get_mla_component_cache_capability(vllm_config) is not None


def _typed_empty_like_storage(raw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bind ``raw``'s storage to a typed one-dimensional view."""
    # raw int8 storage → 目标dtype typed view（一维的element长度的dtype类型的view）
    raw_byte_offset = raw.storage_offset()
    dtype_size = torch.empty((), dtype=dtype).element_size()
    if raw_byte_offset % dtype_size != 0 or raw.numel() % dtype_size != 0:
        raise ValueError(
            f"Raw cache byte range [{raw_byte_offset}, {raw_byte_offset + raw.numel()}) is not aligned to dtype {dtype}"
        )

    typed = torch.empty(0, dtype=dtype, device=raw.device)
    typed.set_(
        raw.untyped_storage(),
        raw_byte_offset // dtype_size,
        (raw.numel() // dtype_size,),
        (1,),
    )
    return typed


def _component_views(
    typed_raw: torch.Tensor,
    *,
    num_blocks: int,
    kernel_blocks_per_manager: int,
    kernel_block_size: int,
    num_kv_heads: int,
    nope_dim: int,
    rope_dim: int,
    slot_bytes: int,
    nope_slot_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct 4D first-axis-strided nope/rope views over kernel slots."""
    element_size = typed_raw.element_size()
    slot_elements = slot_bytes // element_size
    nope = torch.as_strided(
        typed_raw,
        size=(num_blocks * kernel_blocks_per_manager, kernel_block_size, num_kv_heads, nope_dim),
        stride=(slot_elements, num_kv_heads * nope_dim, nope_dim, 1),
        storage_offset=typed_raw.storage_offset(),
    )
    rope = torch.as_strided(
        typed_raw,
        size=(num_blocks * kernel_blocks_per_manager, kernel_block_size, num_kv_heads, rope_dim),
        stride=(slot_elements, num_kv_heads * rope_dim, rope_dim, 1),
        storage_offset=typed_raw.storage_offset() + nope_slot_bytes // element_size,
    )
    return nope, rope


def _validate_component_pair_tensors(nope: torch.Tensor, rope: torch.Tensor) -> bool:
    """Recognize a dynamically laid-out ``(nope, rope)`` MLA cache pair."""
    if nope.ndim != 4 or rope.ndim != 4 or nope.dtype != rope.dtype:
        return False
    if nope.device != rope.device or nope.shape[:3] != rope.shape[:3]:
        return False
    if any(dim <= 0 for dim in (*nope.shape[:3], nope.shape[3], rope.shape[3])):
        return False

    element_size = nope.element_size()
    slot_bytes = nope.stride(0) * element_size
    if slot_bytes <= 0 or slot_bytes != rope.stride(0) * element_size:
        return False

    nope_strides = (
        slot_bytes // element_size,
        nope.shape[2] * nope.shape[3],
        nope.shape[3],
        1,
    )
    rope_strides = (
        slot_bytes // element_size,
        rope.shape[2] * rope.shape[3],
        rope.shape[3],
        1,
    )
    if nope.stride() != nope_strides or rope.stride() != rope_strides:
        return False

    nope_storage = nope.untyped_storage()
    rope_storage = rope.untyped_storage()
    nope_slot_bytes = nope.shape[1] * nope.shape[2] * nope.shape[3] * element_size
    rope_slot_bytes = rope.shape[1] * rope.shape[2] * rope.shape[3] * element_size
    return (
        nope_storage.data_ptr() == rope_storage.data_ptr()
        and nope_storage.nbytes() == rope_storage.nbytes()
        and rope.data_ptr() == nope.data_ptr() + nope_slot_bytes
        and nope_slot_bytes + rope_slot_bytes <= slot_bytes
    )


def is_mla_component_pair(kv_cache: object) -> bool:
    """Return whether ``kv_cache`` is a component-major MLA tuple."""
    if not isinstance(kv_cache, tuple) or len(kv_cache) != 2:
        return False
    nope, rope = kv_cache
    if not isinstance(nope, torch.Tensor) or not isinstance(rope, torch.Tensor):
        return False
    return _validate_component_pair_tensors(nope, rope)


def _validate_kernel_block_sizes(
    kernel_block_sizes: Sequence[int] | Sequence[Sequence[int]],
    block_size: int,
) -> int:
    # 校验dense路径的kernel_block_sizes：必须是长度为1的序列，且kernel block size等于manager block size。
    # 当前dense路径只能支持单一group的MLA，如deepseek v3
    if len(kernel_block_sizes) != 1:
        raise ValueError(
            f"The dense MLA component cache requires one cache group, but got kernel block sizes {kernel_block_sizes!r}"
        )
    kernel_size = kernel_block_sizes[0]
    if isinstance(kernel_size, Sequence):
        valid = len(kernel_size) == 1 and kernel_size[0] == block_size
        kernel_size = kernel_size[0]
    else:
        valid = kernel_size == block_size
    if not valid:
        raise ValueError(
            "Manager and kernel block sizes must match for the dense MLA component "
            f"cache: manager={block_size}, kernels={kernel_block_sizes!r}"
        )
    return int(kernel_size)


def _unwrap_layer_spec(group_spec: Any, layer_name: str) -> Any:
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return group_spec.kv_cache_specs[layer_name]
    return group_spec


def _descriptor_layer_map(
    kv_cache_config: KVCacheConfig,
) -> dict[str, tuple[KVCacheTensor, int]]:
    result: dict[str, tuple[KVCacheTensor, int]] = {}
    for descriptor in kv_cache_config.kv_cache_tensors:
        for layer_idx, layer_name in enumerate(descriptor.layers):
            if layer_name in result:
                raise ValueError(f"KV cache descriptor repeats layer {layer_name}")
            result[layer_name] = (descriptor, layer_idx)
    return result


def _validate_dense_plan(
    *,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    static_forward_context: Mapping[str, AttentionLayerBase],
) -> tuple[_MLAGeometry, KVCacheTensor, Sequence[str]]:
    """确保以下内容：
    1. 逻辑上确实是单一 dense MLA group；
    2. 每个 layer 的 spec/geometry 能推导出同一个 page size；
    3. descriptor 描述的物理地址和 component-major view 公式一致；
    4. resolved layout 确实是 LBNHC；
    5. 没有 sharing / padding / overlay / multi-backing 等未支持情况。
    """
    if len(kv_cache_config.kv_cache_groups) != 1:
        raise ValueError(
            f"The dense MLA component cache requires exactly one group, got {len(kv_cache_config.kv_cache_groups)}"
        )
    group = kv_cache_config.kv_cache_groups[0]
    group_spec = group.kv_cache_spec
    layer_names = group.layer_names
    if not layer_names:
        raise ValueError("The MLA component cache group has no layers")
    if kv_cache_config.num_blocks <= 0:
        raise ValueError("The MLA component cache requires at least one block")
    if len(kv_cache_config.kv_cache_tensors) != 1:
        raise ValueError(
            "The dense MLA component cache requires one backing descriptor, got "
            f"{len(kv_cache_config.kv_cache_tensors)}"
        )
    descriptor = kv_cache_config.kv_cache_tensors[0]

    geometries: list[_MLAGeometry] = []
    specs: list[MLAAttentionSpec] = []
    for layer_name in layer_names:
        layer = static_forward_context.get(layer_name)
        if not isinstance(layer, MLAAttention):
            raise ValueError(f"MLA component-cache layer {layer_name} is {type(layer).__name__}")
        if getattr(layer, "kv_sharing_target_layer_name", None) is not None:
            raise ValueError(f"MLA component-cache layer {layer_name} uses cross-layer KV sharing")
        layer_spec = layer.get_attn_backend().customize_spec(layer.get_kv_cache_spec(vllm_config))
        geometries.append(_layer_geometry(layer, layer_spec, layer_name))
        specs.append(layer_spec)

    geometry = geometries[0]
    if any(item != geometry for item in geometries[1:]):
        raise ValueError("MLA component-cache layers have different geometry")
    if type(group_spec) is not MLAAttentionSpec or any(spec != group_spec for spec in specs):
        raise ValueError("The MLA cache group does not use one exact upstream spec")

    layer_set = set(layer_names)
    descriptor_layers = list(descriptor.layers)
    static_mla_layers = {
        layer_name for layer_name, layer in static_forward_context.items() if isinstance(layer, MLAAttention)
    }
    if (
        len(layer_names) != len(layer_set)
        or len(descriptor_layers) != len(set(descriptor_layers))
        or set(descriptor_layers) != layer_set
        or static_mla_layers != layer_set
    ):
        raise ValueError("The MLA descriptor and cache-group layers do not match")

    num_blocks = kv_cache_config.num_blocks
    expected_layer_stride = geometry.page_bytes * num_blocks
    if descriptor.offset != 0:
        raise ValueError("The dense MLA component cache requires a zero descriptor offset")
    if descriptor.block_stride != geometry.page_bytes:
        raise ValueError("The MLA component descriptor block stride is inconsistent")
    if descriptor.layer_stride != expected_layer_stride:
        raise ValueError("The MLA component descriptor layer stride is inconsistent")
    if descriptor.size != len(descriptor_layers) * expected_layer_stride:
        raise ValueError("The MLA component descriptor size is inconsistent")
    if kv_cache_config.kv_cache_layout != KVCacheLayout.LBNHC.name:
        raise ValueError(f"The MLA component cache requires the LBNHC layout, got {kv_cache_config.kv_cache_layout!r}")
    return geometry, descriptor, descriptor_layers


def allocate_mla_component_cache(
    *,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    static_forward_context: Mapping[str, AttentionLayerBase],
    device: torch.device,
    kernel_block_sizes: Sequence[int] | Sequence[Sequence[int]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Allocate one dense backing and return per-layer ``(nope, rope)`` views."""
    # 分配一个物理backing，并为每个MLA layer构造首轴非连续的nope/rope view。
    geometry, descriptor, layer_names = _validate_dense_plan(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        static_forward_context=static_forward_context,
    )
    kernel_block_size = _validate_kernel_block_sizes(kernel_block_sizes, geometry.block_size)
    if kernel_block_size != geometry.block_size:
        raise ValueError("The dense MLA component cache does not support kernel block splitting")

    raw = torch.zeros(descriptor.size, dtype=torch.int8, device=device)
    typed_raw = _typed_empty_like_storage(raw, geometry.dtype)
    num_blocks = kv_cache_config.num_blocks
    kv_caches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx, layer_name in enumerate(layer_names):
        layer_base_bytes = layer_idx * descriptor.layer_stride
        layer_typed = typed_raw.narrow(
            0,
            layer_base_bytes // geometry.element_size,
            descriptor.layer_stride // geometry.element_size,
        )
        nope, rope = _component_views(
            layer_typed,
            num_blocks=num_blocks,
            kernel_blocks_per_manager=1,
            kernel_block_size=geometry.block_size,
            num_kv_heads=geometry.num_kv_heads,
            nope_dim=geometry.nope_dim,
            rope_dim=geometry.rope_dim,
            slot_bytes=geometry.page_bytes,
            nope_slot_bytes=geometry.nope_segment_bytes,
        )
        if not is_mla_component_pair((nope, rope)):
            raise ValueError("Failed to construct the dense MLA component-cache views")
        kv_caches[layer_name] = (nope, rope)
    return kv_caches


def _validate_hybrid_plan_structure(
    *,
    kv_cache_config: KVCacheConfig,
    static_forward_context: Mapping[str, AttentionLayerBase],
    raw_kv_cache_tensors: Mapping[str, torch.Tensor],
) -> tuple[dict[str, tuple[KVCacheTensor, int]], int]:
    """Validate group/descriptor layer coverage and the one HMA backing."""
    descriptor_layers = _descriptor_layer_map(kv_cache_config)
    persistent_group_layers: set[str] = set()

    for group in kv_cache_config.kv_cache_groups:
        if not group.layer_names:
            raise ValueError("The K3 hybrid KV cache plan contains an empty group")
        representative_spec = _unwrap_layer_spec(group.kv_cache_spec, group.layer_names[0])
        if isinstance(representative_spec, EncoderOnlyAttentionSpec):
            continue

        for layer_name in group.layer_names:
            layer_spec = _unwrap_layer_spec(group.kv_cache_spec, layer_name)
            layer = static_forward_context.get(layer_name)
            if isinstance(layer, MLAAttention):
                if type(layer_spec) is not MLAAttentionSpec:
                    raise ValueError(f"Hybrid MLA layer {layer_name} does not use an exact upstream spec")
            elif not isinstance(layer_spec, MambaSpec):
                raise ValueError(
                    f"Hybrid non-MLA layer {layer_name} uses {type(layer_spec).__name__}; "
                    "only MambaSpec/KDA is supported"
                )
            persistent_group_layers.add(layer_name)

    if persistent_group_layers != set(descriptor_layers):
        missing = sorted(persistent_group_layers - set(descriptor_layers))
        extra = sorted(set(descriptor_layers) - persistent_group_layers)
        raise ValueError(f"Hybrid KV cache group/descriptor layer mismatch: missing={missing}, extra={extra}")

    if set(raw_kv_cache_tensors) != persistent_group_layers:
        missing = sorted(persistent_group_layers - set(raw_kv_cache_tensors))
        extra = sorted(set(raw_kv_cache_tensors) - persistent_group_layers)
        raise ValueError(f"Hybrid raw tensor layer mismatch: missing={missing}, extra={extra}")

    descriptor_sizes = {descriptor.size for descriptor in kv_cache_config.kv_cache_tensors}
    if len(descriptor_sizes) != 1:
        raise ValueError(f"Hybrid KV cache descriptors must share one backing size, got {sorted(descriptor_sizes)}")
    backing_size = next(iter(descriptor_sizes))

    storage_ptrs: set[int] = set()
    for layer_name, raw in raw_kv_cache_tensors.items():
        if not isinstance(raw, torch.Tensor):
            raise ValueError(f"Hybrid raw cache for {layer_name} is not a tensor")
        storage_ptrs.add(raw.untyped_storage().data_ptr())
    if len(storage_ptrs) != 1:
        raise ValueError("The K3 hybrid component cache requires one standardized HMA backing allocation")

    return descriptor_layers, backing_size


def materialize_hybrid_mla_component_cache(
    *,
    raw_kv_cache_tensors: Mapping[str, torch.Tensor],
    kv_cache_config: KVCacheConfig,
    static_forward_context: Mapping[str, AttentionLayerBase],
    kernel_block_sizes: Sequence[int] | Sequence[Sequence[int]],
    capability: MLAComponentCacheCapability,
    vllm_config: VllmConfig,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Convert hybrid MLA raw layer regions into component-major tuple views.

    The legacy allocator has already materialized all standardized descriptor
    regions.  This function replaces only the MLA entries; Mamba/KDA raw tensors
    remain untouched for the legacy reshape path.

    K3 hybrid复用standardized allocator生成的HMA raw region：
    1. 只把MLA raw region重排成component-major tuple；
    2. KDA/Mamba raw region保持旧路径语义；
    3. 所有persistent region必须来自同一个backing。
    """
    if kv_cache_config.kv_cache_layout != KVCacheLayout.LBNHC.name:
        raise ValueError(
            f"The K3 hybrid MLA component cache requires the LBNHC layout, got {kv_cache_config.kv_cache_layout!r}"
        )
    descriptor_layers, backing_size = _validate_hybrid_plan_structure(
        kv_cache_config=kv_cache_config,
        static_forward_context=static_forward_context,
        raw_kv_cache_tensors=raw_kv_cache_tensors,
    )
    component_caches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    mla_layer_set = set(capability.mla_layer_names)
    persistent_group_idx = -1

    for group in kv_cache_config.kv_cache_groups:
        representative_spec = _unwrap_layer_spec(group.kv_cache_spec, group.layer_names[0])
        if isinstance(representative_spec, EncoderOnlyAttentionSpec):
            continue
        persistent_group_idx += 1
        if persistent_group_idx >= len(kernel_block_sizes):
            raise ValueError(f"Missing kernel block size for KV cache group {persistent_group_idx}")
        selected = kernel_block_sizes[persistent_group_idx]
        kernel_block_size = selected[0] if isinstance(selected, Sequence) else selected

        for layer_name in group.layer_names:
            layer = static_forward_context.get(layer_name)
            if not isinstance(layer, MLAAttention):
                continue

            group_spec = _unwrap_layer_spec(group.kv_cache_spec, layer_name)
            if type(group_spec) is not MLAAttentionSpec:
                raise ValueError(f"Hybrid MLA layer {layer_name} does not use an exact upstream spec")
            if layer_name not in descriptor_layers:
                raise ValueError(f"Hybrid MLA layer {layer_name} has no descriptor")
            descriptor, layer_idx = descriptor_layers[layer_name]
            raw = raw_kv_cache_tensors.get(layer_name)
            if not isinstance(raw, torch.Tensor):
                raise ValueError(f"Hybrid MLA layer {layer_name} has no raw backing slice")

            logical_spec = layer.get_attn_backend().customize_spec(layer.get_kv_cache_spec(vllm_config))
            geometry = _layer_geometry(layer, logical_spec, layer_name)
            if (
                capability.manager_block_size != geometry.block_size
                or geometry.block_size != group_spec.block_size
                or geometry.dtype != group_spec.dtype
                or geometry.num_kv_heads != group_spec.num_kv_heads
                or geometry.nope_dim + geometry.rope_dim != group_spec.head_size
            ):
                raise ValueError(f"Hybrid MLA layer {layer_name} disagrees with its merged spec or capability")
            if kernel_block_size != capability.kernel_block_size:
                raise ValueError(
                    f"Hybrid MLA layer {layer_name} kernel size {kernel_block_size} does not match capability "
                    f"{capability.kernel_block_size}"
                )
            if geometry.block_size % kernel_block_size != 0:
                raise ValueError(
                    f"Hybrid MLA manager block size {geometry.block_size} is not divisible by "
                    f"kernel size {kernel_block_size}"
                )

            num_blocks = kv_cache_config.num_blocks
            physical_page_bytes = group_spec.page_size_bytes
            real_manager_page_bytes = (
                geometry.block_size
                * geometry.num_kv_heads
                * (geometry.nope_dim + geometry.rope_dim)
                * geometry.element_size
            )
            if physical_page_bytes < real_manager_page_bytes:
                raise ValueError(
                    f"Hybrid MLA physical page {physical_page_bytes} is smaller than "
                    f"real page {real_manager_page_bytes}"
                )
            if physical_page_bytes % geometry.element_size != 0:
                raise ValueError(f"Hybrid MLA physical page {physical_page_bytes} is not dtype-aligned")
            ratio = geometry.block_size // kernel_block_size
            if physical_page_bytes % ratio != 0:
                raise ValueError(
                    f"Hybrid MLA physical page {physical_page_bytes} cannot be divided into {ratio} kernel slots"
                )
            slot_bytes = physical_page_bytes // ratio
            if slot_bytes % geometry.element_size != 0:
                raise ValueError(f"Hybrid MLA kernel slot {slot_bytes} is not dtype-aligned")
            nope_slot_bytes = kernel_block_size * geometry.num_kv_heads * geometry.nope_dim * geometry.element_size
            rope_slot_bytes = kernel_block_size * geometry.num_kv_heads * geometry.rope_dim * geometry.element_size
            if nope_slot_bytes + rope_slot_bytes > slot_bytes:
                raise ValueError(
                    f"Hybrid MLA kernel slot {slot_bytes} is smaller than nope+rope {nope_slot_bytes + rope_slot_bytes}"
                )

            expected_layer_stride = physical_page_bytes * num_blocks
            if descriptor.block_stride != physical_page_bytes:
                raise ValueError(f"Hybrid MLA descriptor block stride is inconsistent for {layer_name}")
            if descriptor.layer_stride != expected_layer_stride:
                raise ValueError(f"Hybrid MLA descriptor layer stride is inconsistent for {layer_name}")
            expected_offset = descriptor.offset + layer_idx * descriptor.layer_stride
            if raw.storage_offset() != expected_offset:
                raise ValueError(f"Hybrid MLA raw offset is inconsistent for {layer_name}")
            if raw.numel() != expected_layer_stride:
                raise ValueError(f"Hybrid MLA raw size is inconsistent for {layer_name}")
            if descriptor.size != backing_size or expected_offset + expected_layer_stride > backing_size:
                raise ValueError(f"Hybrid MLA descriptor range exceeds backing for {layer_name}")

            typed_raw = _typed_empty_like_storage(raw, geometry.dtype)
            nope, rope = _component_views(
                typed_raw,
                num_blocks=num_blocks,
                kernel_blocks_per_manager=ratio,
                kernel_block_size=kernel_block_size,
                num_kv_heads=geometry.num_kv_heads,
                nope_dim=geometry.nope_dim,
                rope_dim=geometry.rope_dim,
                slot_bytes=slot_bytes,
                nope_slot_bytes=nope_slot_bytes,
            )
            if not is_mla_component_pair((nope, rope)):
                raise ValueError(f"Failed to construct hybrid MLA component-cache views for {layer_name}")
            component_caches[layer_name] = (nope, rope)

    if set(component_caches) != mla_layer_set:
        missing = sorted(mla_layer_set - set(component_caches))
        extra = sorted(set(component_caches) - mla_layer_set)
        raise ValueError(f"Hybrid MLA layer mismatch: missing={missing}, extra={extra}")

    return component_caches
