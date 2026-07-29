# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/attn_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from math import prod
from typing import Any

import numpy as np
import torch
import vllm
from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup, extract_layer_index

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.quantization.utils import enable_fa_quant
from vllm_ascend.utils import calc_split_factor, vllm_version_is

_ATTENTION_MASK_BUILDER = None


def bind_kv_cache(
    kv_caches: dict[
        str,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ],
    forward_context: dict[str, AttentionLayerBase],
    runner_kv_caches: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    num_attn_module: int = 1,
) -> None:
    """Bind all Ascend attention cache components to ModelRunner V2.

    A single transformer layer may expose multiple cache-only attention
    modules.  This is normal for backends such as DSA (SWA, compressor and
    indexer caches), and future plugin backends can use the same layout.
    Upstream currently rejects this case for non CUDA/XPU/CPU platforms even
    though the generic binding itself is valid on NPU.
    """
    if runner_kv_caches:
        raise ValueError("ModelRunner KV caches must be empty before binding.")

    index_to_names: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        layer_index = extract_layer_index(layer_name, num_attn_module)
        index_to_names[layer_index].append(layer_name)

    for layer_index in sorted(index_to_names):
        for layer_name in index_to_names[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        if layer_name not in forward_context:
            raise KeyError(f"KV cache layer {layer_name} is missing from forward context.")
        forward_context[layer_name].kv_cache = kv_cache


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """Build Ascend-specific KV cache specs for v2 worker patching."""
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    layer_type = AttentionLayerBase
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)

    for layer_name, attn_module in attn_layers.items():
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
            continue
        if isinstance(attn_module, Attention):
            if spec := attn_module.get_kv_cache_spec(vllm_config):
                kv_cache_spec[layer_name] = spec
            continue
        if isinstance(attn_module, MLAAttention):
            spec = attn_module.get_kv_cache_spec(vllm_config)
            if spec is None:
                continue
            if getattr(attn_module.impl, "fa_quant_layer", False):
                head_size = attn_module.head_size + attn_module.qk_rope_head_dim
                dtype, cache_dtype_str = attn_module.impl.dtype, None
            else:
                head_size = spec.head_size
                dtype = spec.dtype
                cache_dtype_str = spec.cache_dtype_str
            kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                cache_dtype_str=cache_dtype_str,
            )
            continue

        # Cache-only attention modules (for example DeepSeek-V4's SWA,
        # compressor-state and indexer caches) intentionally implement the
        # AttentionLayerBase contract without inheriting Attention or
        # MLAAttention.  Keep the special conversions above for the two generic
        # layer types, then fall back to the backend-neutral cache-spec hook.
        # This also lets future plugin attention backends participate in MRV2
        # without growing another isinstance branch here.
        get_kv_cache_spec_fn = getattr(attn_module, "get_kv_cache_spec", None)
        if get_kv_cache_spec_fn is not None:
            if spec := get_kv_cache_spec_fn(vllm_config):
                kv_cache_spec[layer_name] = spec

    return kv_cache_spec


def get_attn_mask_builder(device: torch.device):
    """Get attention mask builder which only have one instance."""
    global _ATTENTION_MASK_BUILDER
    if _ATTENTION_MASK_BUILDER is None:
        _ATTENTION_MASK_BUILDER = AttentionMaskBuilder(device)
    return _ATTENTION_MASK_BUILDER


def build_attn_metadata(
    *,
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    dcp_local_seq_lens: torch.Tensor | None = None,
    # extra attributes for ascend npus.
    seq_lens_np: np.ndarray | None = None,
    seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    num_computed_tokens_cpu: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    attn_state: Any | None = None,
    graph_pad_size: int = -1,
    num_input_tokens: int = 0,
    model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    for_cudagraph_capture: bool = False,
    causal: bool | Mapping[int, bool] = True,
) -> dict[str, Any]:
    """Build attention metadata for Ascend NPUs."""
    # TODO(Ronald1995): optimize AscendCommonAttentionMetadata.

    # seq_lens_np is used for ascend npus, it maybe None in spec_decode case,
    # we fill it with max_seq_len in case `attn_metadata_builder.build` raise
    # an error.
    if seq_lens_np is None:
        seq_lens_np = np.full(num_reqs, max_seq_len, dtype=np.int32)
    seq_lens_cpu = torch.from_numpy(seq_lens_np)[:num_reqs]
    if not vllm_version_is("0.25.1") and seq_lens_cpu_upper_bound is None:
        seq_lens_cpu_upper_bound = seq_lens_cpu

    attn_metadata: dict[str, Any] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups
    for i, kv_cache_spec in enumerate(kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]
        # Hybrid drafters can configure causality per KV cache group.
        group_causal = causal if isinstance(causal, bool) else causal.get(i, True)

        common_attn_metadata_extra_kwargs = (
            model_specific_attn_metadata.get_extra_common_attn_kwargs(i, num_reqs)
            if model_specific_attn_metadata is not None
            else {}
        )
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            seq_lens=seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            positions=positions,
            attn_state=attn_state,
            graph_pad_size=graph_pad_size,
            num_input_tokens=num_input_tokens,
            max_seq_len=max_seq_len,
            causal=group_causal,
            **common_attn_metadata_extra_kwargs,
        )

        for attn_group in attn_groups[i]:
            attn_metadata_builder = attn_group.get_metadata_builder(0)
            if for_cudagraph_capture:
                metadata = attn_metadata_builder.build_for_cudagraph_capture(common_attn_metadata)
            else:
                attn_metadata_extra_kwargs = (
                    model_specific_attn_metadata.get_extra_attn_kwargs(
                        attn_metadata_builder,
                        num_reqs,
                    )
                    if model_specific_attn_metadata is not None
                    else {}
                )
                metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
    return attn_metadata


def build_attn_state(
    vllm_config: VllmConfig,
    seq_lens_np: np.ndarray,
    num_reqs,
    num_scheduled_tokens,
    num_valid_tokens,
):
    """Build attention state for npu's attention backend."""
    if vllm_config.model_config.runner_type == "pooling":
        if isinstance(
            vllm_config.kv_cache_config.kv_cache_groups[0].kv_cache_spec,
            EncoderOnlyAttentionSpec,
        ):
            attn_state = AscendAttentionState.PrefillNoCache
        else:
            attn_state = AscendAttentionState.PrefillCacheHit
    elif np.array_equal(seq_lens_np[:num_reqs], num_scheduled_tokens):
        attn_state = AscendAttentionState.PrefillNoCache
    # We assume it is the decode stage, where prefill occurs
    # but only one token is not hit in cache.
    elif np.all(num_scheduled_tokens == 1):
        attn_state = AscendAttentionState.DecodeOnly
        if vllm_config.speculative_config and vllm_config.speculative_config.method == "mtp":
            # SpecDecoding now supports seq_len=1 and seq_len=2
            # In Prefilling Decoding Disaggregation scenario, SpecDecoding
            # need to supports seq_len=1
            attn_state = AscendAttentionState.SpecDecoding
    # Speculative decoding.
    elif np.all(num_valid_tokens == 1):
        if vllm_config.speculative_config and vllm_config.speculative_config.method == "mtp":
            attn_state = AscendAttentionState.SpecDecoding
        else:
            attn_state = AscendAttentionState.ChunkedPrefill
    # splitfuse
    elif vllm_config.scheduler_config.enable_chunked_prefill:
        attn_state = AscendAttentionState.ChunkedPrefill
    else:
        attn_state = AscendAttentionState.PrefillCacheHit
    return attn_state


def _get_layer_kv_cache_specs(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    layer_kv_cache_spec: dict[str, KVCacheSpec] = {}
    for group_kv_cache_spec in kv_cache_config.kv_cache_groups:
        group_spec = group_kv_cache_spec.kv_cache_spec
        for layer_name in group_kv_cache_spec.layer_names:
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec[layer_name] = group_spec.kv_cache_specs[layer_name]
            else:
                layer_kv_cache_spec[layer_name] = group_spec
    return layer_kv_cache_spec


def _get_attention_kv_cache_dims(layer_name: str, kv_cache_spec: AttentionSpec) -> tuple[int, int]:
    if isinstance(kv_cache_spec, AscendMLAAttentionSpec):
        attn_layers = get_layers_from_vllm_config(get_current_vllm_config(), AttentionLayerBase, [layer_name])
        attn_layer = attn_layers[layer_name]
        if not isinstance(attn_layer, MLAAttention):
            raise TypeError(f"Expected AscendMLAAttention layer for {layer_name}, got {type(attn_layer).__name__}.")
        return attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim

    head_size_v = kv_cache_spec.head_size_v if hasattr(kv_cache_spec, "head_size_v") else kv_cache_spec.head_size
    return kv_cache_spec.head_size, head_size_v


def _get_attention_kv_cache_component_layout(
    layer_name: str,
    kv_cache_spec: AttentionSpec,
) -> tuple[tuple[int, torch.dtype], ...]:
    """Return physical ``(last_dim, dtype)`` components for one cache layer.

    Ascend historically split every cache into K/V tensors for KV transfer.
    Plugin attention backends can override that assumption through
    ``get_kv_cache_component_layout``.  This keeps allocation backend-neutral
    and lets combined-vector or data/scale caches describe their real layout.
    """
    attn_layers = get_layers_from_vllm_config(get_current_vllm_config(), AttentionLayerBase, [layer_name])
    attn_layer = attn_layers[layer_name]
    backend = attn_layer.get_attn_backend()
    layout_hook = getattr(backend, "get_kv_cache_component_layout", None)
    if layout_hook is not None:
        layout = tuple(layout_hook(kv_cache_spec))
    else:
        k_dim, v_dim = _get_attention_kv_cache_dims(layer_name, kv_cache_spec)
        k_dtype = v_dtype = kv_cache_spec.dtype
        vllm_config = get_current_vllm_config()
        if enable_fa_quant(vllm_config, layer_name):
            k_dtype, v_dtype = vllm_config.quant_config.get_kv_quant_dtype(
                layer_name,
                kv_cache_spec.dtype,
                vllm_config.model_config,
            )
        layout = ((k_dim, k_dtype), (v_dim, v_dtype))

    if not 1 <= len(layout) <= 2:
        raise ValueError(
            f"MRV2 Ascend cache layout for {layer_name} must contain one or two components, got {len(layout)}."
        )
    if any(dim <= 0 for dim, _ in layout):
        raise ValueError(f"MRV2 Ascend cache layout for {layer_name} has a non-positive component dimension: {layout}.")
    return layout


def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset) :]


def _view_cache_component(
    raw_tensor: torch.Tensor,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    page_size_bytes: int,
) -> torch.Tensor:
    """View one raw cache component, preserving padded page strides."""
    dtype_size = get_dtype_size(dtype)
    required_bytes = prod(shape) * dtype_size
    if raw_tensor.numel() == required_bytes:
        return raw_tensor.view(dtype).view(shape)

    if raw_tensor.numel() % page_size_bytes != 0:
        raise ValueError(f"Cache allocation size {raw_tensor.numel()} is not divisible by page size {page_size_bytes}.")
    num_pages = raw_tensor.numel() // page_size_bytes
    if not shape or shape[0] != num_pages:
        raise ValueError(
            f"Only a num-pages-first cache layout can use padded pages: shape={shape}, num_pages={num_pages}."
        )
    if page_size_bytes % dtype_size:
        raise ValueError(f"Cache page size {page_size_bytes} is not aligned to {dtype}.")

    strides = list(torch.empty(shape).stride())
    strides[0] = page_size_bytes // dtype_size
    return torch.as_strided(
        raw_tensor.view(dtype),
        size=shape,
        stride=tuple(strides),
    )


def _reshape_attention_cache(
    layer_name: str,
    kv_cache_spec: AttentionSpec,
    raw_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_cache_shape: tuple[int, ...],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    component_layout = _get_attention_kv_cache_component_layout(layer_name, kv_cache_spec)
    if len(component_layout) == 1:
        if not isinstance(raw_cache, torch.Tensor):
            raise TypeError(f"Cache layer {layer_name} declares a single component but received a split allocation.")
        dim, dtype = component_layout[0]
        shape = (*kv_cache_shape[:-1], dim)
        return _view_cache_component(
            raw_cache,
            dtype,
            shape,
            kv_cache_spec.page_size_bytes,
        )

    if not isinstance(raw_cache, tuple):
        raise TypeError(f"Cache layer {layer_name} declares two components but received a single allocation.")
    if isinstance(kv_cache_spec, (AscendMLAAttentionSpec, MLAAttentionSpec)):
        component_prefix = kv_cache_shape[:-1]
    else:
        # Conventional attention backends include the K/V axis first.
        if not kv_cache_shape or kv_cache_shape[0] != 2:
            raise ValueError(f"Expected a leading K/V axis for {layer_name}, got shape {kv_cache_shape}.")
        component_prefix = kv_cache_shape[1:-1]

    components = []
    for raw_tensor, (dim, dtype) in zip(raw_cache, component_layout):
        components.append(
            _view_cache_component(
                raw_tensor,
                dtype,
                (*component_prefix, dim),
                kv_cache_spec.page_size_bytes,
            )
        )
    return components[0], components[1]


def _allocate_kv_cache(
    kv_cache_config: KVCacheConfig,
    shared_layers: dict[str, str],
    device: torch.device,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    """
    Initialize the KV cache buffer with the correct size. The buffer needs to be
    reshaped to the desired shape before being used by the models.

    NOTE: To support prefill disaggregation, we need to split kvcache tensor
    into k_cache and v_cache, and the addr of both are aligned by 2M.

    Args:
        kv_cache_config: The KV cache config
        device: The device
    Returns:
        dict[str, tuple[torch.Tensor, torch.Tensor]]: A map between layer names
            to their corresponding memory buffer for K cache and V cache
    """
    vllm_config = get_current_vllm_config()

    # init kv cache tensors
    kv_cache_raw_tensors: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
    # prefill disaggregation need the addr of cache tensor be aligned with 2M
    alignment = 2 * 1024 * 1024
    layer_kv_cache_spec = _get_layer_kv_cache_specs(kv_cache_config)
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if len(kv_cache_tensor.shared_by) == 0:
            continue

        layer_layouts: dict[str, tuple[tuple[int, torch.dtype], ...]] = {}
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_spec = layer_kv_cache_spec[layer_name]
            assert isinstance(kv_cache_spec, AttentionSpec)
            layer_layouts[layer_name] = _get_attention_kv_cache_component_layout(layer_name, kv_cache_spec)

        if vllm_config.kv_transfer_config is None:
            # A scheduler KVCacheTensor may be reused by heterogeneous cache
            # layers.  Keep one physical allocation and derive a layout-specific
            # raw view for every layer instead of imposing the first layer's
            # K/V structure on all aliases.
            backing = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
            for layer_name, component_layout in layer_layouts.items():
                if len(component_layout) == 1:
                    kv_cache_raw_tensors[layer_name] = backing
                    continue
                component_bytes = [dim * get_dtype_size(dtype) for dim, dtype in component_layout]
                split_factors = calc_split_factor(component_bytes)
                first_size = int(kv_cache_tensor.size // split_factors[0])
                second_size = kv_cache_tensor.size - first_size
                kv_cache_raw_tensors[layer_name] = (
                    backing[:first_size],
                    backing[first_size : first_size + second_size],
                )
            continue

        # KV transfer requires independently aligned component addresses.
        # A shared physical allocation therefore cannot represent aliases with
        # different component structures in that mode.
        unique_layouts = set(layer_layouts.values())
        if len(unique_layouts) != 1:
            raise ValueError(
                "KV transfer does not support a shared cache allocation with "
                f"heterogeneous component layouts: {layer_layouts}."
            )
        component_layout = next(iter(unique_layouts))
        if len(component_layout) == 1:
            raw_tensor = torch.zeros(
                kv_cache_tensor.size + alignment,
                dtype=torch.int8,
                device=device,
            )
            cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = _align_memory(raw_tensor, alignment)[
                : kv_cache_tensor.size
            ]
        else:
            component_bytes = [dim * get_dtype_size(dtype) for dim, dtype in component_layout]
            split_factors = calc_split_factor(component_bytes)
            first_size = int(kv_cache_tensor.size // split_factors[0])
            second_size = kv_cache_tensor.size - first_size
            first_tensor = torch.zeros(
                first_size + alignment,
                dtype=torch.int8,
                device=device,
            )
            second_tensor = torch.zeros(
                second_size + alignment,
                dtype=torch.int8,
                device=device,
            )
            cache = (
                _align_memory(first_tensor, alignment)[:first_size],
                _align_memory(second_tensor, alignment)[:second_size],
            )
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = cache

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            layer_names.add(layer_name)
    assert layer_names == (kv_cache_raw_tensors.keys() | shared_layers.keys()), (
        "Some layers are not correctly initialized"
    )

    return kv_cache_raw_tensors


def _reshape_kv_cache(
    kv_cache_config: KVCacheConfig,
    kv_cache_raw_tensors: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    attn_backends: dict[str, AttentionBackend],
    cache_dtype: str,
    kernel_block_sizes: list[int] | None = None,
    shared_kv_cache_layers: dict[str, str] | None = None,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    """
    Reshape the KV cache tensors to the desired shape and dtype.

    Args:
        kv_cache_config: The KV cache config
        kv_cache_raw_tensors: The KV cache buffer of each layer, with correct
            size but uninitialized shape
    Returns:
        dict[str, tuple[torch.Tensor, torch.Tensor]]: A map between layer names
            to their corresponding memory buffer for KV cache
    """
    kv_caches: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
    kernel_block_sizes = kernel_block_sizes or []
    for kv_cache_group_id, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
        for layer_name in kv_cache_group_spec.layer_names:
            if shared_kv_cache_layers and layer_name in shared_kv_cache_layers:
                continue
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[layer_name]
            assert isinstance(kv_cache_spec, AttentionSpec)

            if isinstance(kv_cache_spec, AttentionSpec):
                raw_cache = kv_cache_raw_tensors[layer_name]
                sum_page_size_bytes = (
                    raw_cache.numel()
                    if isinstance(raw_cache, torch.Tensor)
                    else sum(tensor.numel() for tensor in raw_cache)
                )
                assert sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
                num_blocks = sum_page_size_bytes // kv_cache_spec.page_size_bytes

                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks

                attn_backend = attn_backends[layer_name]
                if kv_cache_group_id < len(kernel_block_sizes):
                    kernel_block_size = kernel_block_sizes[kv_cache_group_id]
                    num_blocks *= kv_cache_spec.block_size // kernel_block_size
                else:
                    kernel_block_size = kv_cache_spec.block_size

                if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
                    shape_block_size = kv_cache_spec.storage_block_size
                else:
                    shape_block_size = kernel_block_size

                kv_cache_shape = attn_backend.get_kv_cache_shape(
                    num_blocks,
                    shape_block_size,
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                    cache_dtype,
                )
                kv_caches[layer_name] = _reshape_attention_cache(
                    layer_name,
                    kv_cache_spec,
                    raw_cache,
                    kv_cache_shape,
                )
            else:
                raise ValueError("Unknown KV cache spec type.")

    if shared_kv_cache_layers:
        for layer_name, target_layer_name in shared_kv_cache_layers.items():
            kv_caches[layer_name] = kv_caches[target_layer_name]

    return kv_caches


def _reshape_kv_cache_v2(
    attn_groups: Sequence[AttentionGroup],
    kv_cache_raw_tensors: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    cache_dtype: str,
    kernel_block_sizes: list[int],
    shared_kv_cache_layers: dict[str, str],
    kv_cache_config: "KVCacheConfig | None" = None,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    kv_caches: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
    for group in attn_groups:
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue

        kv_cache_spec = group.kv_cache_spec
        if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
            kernel_block_size = kv_cache_spec.storage_block_size
        else:
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]

        for layer_name in group.layer_names:
            if layer_name in shared_kv_cache_layers:
                continue

            assert isinstance(kv_cache_spec, AttentionSpec)

            raw_cache = kv_cache_raw_tensors[layer_name]
            sum_page_size_bytes = (
                raw_cache.numel()
                if isinstance(raw_cache, torch.Tensor)
                else sum(tensor.numel() for tensor in raw_cache)
            )
            assert sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
            num_blocks = sum_page_size_bytes // kv_cache_spec.page_size_bytes

            num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
            kernel_num_blocks = num_blocks * num_blocks_per_kv_block

            kv_cache_shape = group.backend.get_kv_cache_shape(
                kernel_num_blocks,
                kernel_block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype,
            )

            kv_caches[layer_name] = _reshape_attention_cache(
                layer_name,
                kv_cache_spec,
                raw_cache,
                kv_cache_shape,
            )

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]

    return kv_caches


_BUILD_ATTN_METADATA_MODULE = vllm.v1.worker.gpu.spec_decode.speculator


@contextmanager
def build_attn_metadata_wrapper():
    """Context manager to override attention metadata building for Ascend NPUs."""
    original_func = _BUILD_ATTN_METADATA_MODULE.build_attn_metadata
    try:
        _BUILD_ATTN_METADATA_MODULE.build_attn_metadata = build_attn_metadata
        yield
    finally:
        _BUILD_ATTN_METADATA_MODULE.build_attn_metadata = original_func
