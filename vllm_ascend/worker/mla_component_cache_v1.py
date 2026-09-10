# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The vllm-ascend contributors
"""Allocation and validation of the V1 dense-MLA component-major cache."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheLayout,
    KVCacheTensor,
    KVQuantMode,
    MLAAttentionSpec,
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


def _feature_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE)


def _unsupported(reason: str) -> ValueError:
    return ValueError(
        "VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE=1 is enabled, but the model "
        f"does not satisfy the dense-MLA component-cache capability: {reason}"
    )


def _validate_exact_spec(spec: Any, layer_name: str) -> None:
    # 没有任何扩展、没有量化、没有压缩、没有 padding、没有 sliding window 的 exact upstream MLAAttentionSpec
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


def use_mla_component_cache(vllm_config: VllmConfig) -> bool:
    """Return whether the runner can use the V1 component-major MLA cache.

    The feature is intentionally explicit.  When it is enabled, an MLA model
    outside the documented dense capability fails fast instead of silently
    falling back to the legacy Ascend MLA spec path.
    判断当前是否使用MLA首轴非连续能力。
    在 feature 开启时,判断当前是否是“单一、均匀、dense、可 stride 寻址的 MLA cache group”
    满足则返回 True 走 component-major nope/rope 布局
    不满足则启动阶段直接失败，避免静默 fallback 到不兼容路径
    """
    if not _feature_enabled():
        return False

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
    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    mla_layers: dict[str, MLAAttention] = {}
    for layer_name, layer in layers.items():
        if getattr(layer, "kv_sharing_target_layer_name", None) is not None:
            raise _unsupported(f"layer {layer_name} uses cross-layer KV sharing")
        if isinstance(layer, MLAAttention):
            mla_layers[layer_name] = layer
            continue
        try:
            has_non_mla_cache = layer.get_kv_cache_spec(vllm_config) is not None
        except Exception as exc:
            raise _unsupported(f"cannot inspect the KV cache spec of non-MLA layer {layer_name}") from exc
        if has_non_mla_cache:
            raise _unsupported(
                f"layer {layer_name} is {type(layer).__name__}; the component "
                "cache only supports one dense MLA cache group"
            )

    if not mla_layers:
        raise _unsupported("the model has no MLA layers")
    if not isinstance(forward_context, Mapping):
        raise _unsupported("the static forward context is not a mapping")

    first_geometry: _MLAGeometry | None = None
    first_spec: MLAAttentionSpec | None = None
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
        supported_block_sizes = backend.get_supported_kernel_block_sizes()
        if geometry.block_size not in supported_block_sizes:
            raise _unsupported(
                f"layer {layer_name} block size {geometry.block_size} is not in "
                f"the backend supported sizes {supported_block_sizes}"
            )

        if first_geometry is None:
            first_geometry = geometry
            first_spec = spec
        elif geometry != first_geometry or spec != first_spec:
            raise _unsupported(f"layer {layer_name} has different MLA cache geometry")

    assert first_geometry is not None and first_spec is not None
    return True


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


def _validate_kernel_block_sizes(
    kernel_block_sizes: Sequence[int] | Sequence[Sequence[int]],
    block_size: int,
) -> None:
    # 校验kernel_block_sizes是否符合要求，必须是一个长度为1的序列，并且必须等于block_size匹配
    # 当前只能支持单一group的MLA，如deepseek v3
    if len(kernel_block_sizes) != 1:
        raise ValueError(
            f"The MLA component cache requires one cache group, but got kernel block sizes {kernel_block_sizes!r}"
        )
    kernel_size = kernel_block_sizes[0]
    if isinstance(kernel_size, Sequence):
        valid = len(kernel_size) == 1 and kernel_size[0] == block_size
    else:
        valid = kernel_size == block_size
    if not valid:
        raise ValueError(
            "Manager and kernel block sizes must match for the MLA component "
            f"cache: manager={block_size}, kernels={kernel_block_sizes!r}"
        )


def _validate_plan(
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
            "The MLA component cache requires exactly one dense MLA group, got "
            f"{len(kv_cache_config.kv_cache_groups)} groups"
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
            f"The MLA component cache requires one backing descriptor, got {len(kv_cache_config.kv_cache_tensors)}"
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
        raise ValueError("The MLA component cache requires a zero descriptor offset")
    if descriptor.block_stride != geometry.page_bytes:
        raise ValueError("The MLA component descriptor block stride is inconsistent")
    if descriptor.layer_stride != expected_layer_stride:
        raise ValueError("The MLA component descriptor layer stride is inconsistent")
    if descriptor.size != len(descriptor_layers) * expected_layer_stride:
        raise ValueError("The MLA component descriptor size is inconsistent")

    layout_name = kv_cache_config.kv_cache_layout
    if layout_name != KVCacheLayout.LBNHC.name:
        raise ValueError(f"The MLA component cache requires the LBNHC layout, got {layout_name!r}")
    return geometry, descriptor, descriptor_layers


def allocate_mla_component_cache(
    *,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    static_forward_context: Mapping[str, AttentionLayerBase],
    device: torch.device,
    kernel_block_sizes: Sequence[int] | Sequence[Sequence[int]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Allocate one backing and return per-layer ``(nope, rope)`` views."""
    geometry, descriptor, layer_names = _validate_plan(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        static_forward_context=static_forward_context,
    )
    _validate_kernel_block_sizes(kernel_block_sizes, geometry.block_size)

    raw = torch.zeros(descriptor.size, dtype=torch.int8, device=device)
    typed_raw = _typed_empty_like_storage(raw, geometry.dtype)
    page_elements = geometry.page_bytes // geometry.element_size
    nope_page_elements = geometry.nope_segment_bytes // geometry.element_size
    rope_page_elements = geometry.rope_segment_bytes // geometry.element_size
    num_blocks = kv_cache_config.num_blocks

    kv_caches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx, layer_name in enumerate(layer_names):
        layer_base_bytes = layer_idx * descriptor.layer_stride
        nope = torch.as_strided(
            typed_raw,
            size=(num_blocks, geometry.block_size, geometry.num_kv_heads, geometry.nope_dim),
            stride=(
                page_elements,
                geometry.num_kv_heads * geometry.nope_dim,
                geometry.nope_dim,
                1,
            ),
            storage_offset=typed_raw.storage_offset() + layer_base_bytes // geometry.element_size,
        )
        rope = torch.as_strided(
            typed_raw,
            size=(num_blocks, geometry.block_size, geometry.num_kv_heads, geometry.rope_dim),
            stride=(
                page_elements,
                geometry.num_kv_heads * geometry.rope_dim,
                geometry.rope_dim,
                1,
            ),
            storage_offset=(
                typed_raw.storage_offset() + (layer_base_bytes + geometry.nope_segment_bytes) // geometry.element_size
            ),
        )

        expected_nope_stride = (
            page_elements,
            geometry.num_kv_heads * geometry.nope_dim,
            geometry.nope_dim,
            1,
        )
        expected_rope_stride = (
            page_elements,
            geometry.num_kv_heads * geometry.rope_dim,
            geometry.rope_dim,
            1,
        )
        if nope.stride() != expected_nope_stride or rope.stride() != expected_rope_stride:
            raise ValueError("Failed to construct the MLA component-cache strides")
        if (
            nope.untyped_storage().data_ptr() != rope.untyped_storage().data_ptr()
            or rope.data_ptr() != nope.data_ptr() + geometry.nope_segment_bytes
        ):
            raise ValueError("Failed to construct the MLA component-cache storage views")
        if nope_page_elements + rope_page_elements != page_elements:
            raise ValueError("MLA component page contains unexpected padding")

        kv_caches[layer_name] = (nope, rope)
    return kv_caches
