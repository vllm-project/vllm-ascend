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

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
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
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.quantization.utils import enable_fa_quant
from vllm_ascend.utils import calc_split_factor

_ATTENTION_MASK_BUILDER = None


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """Build Ascend-specific KV cache specs for v2 worker patching."""
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    attention_layer_names: list[str] = []
    mamba_specs: dict[str, MambaSpec] = {}
    layer_type = AttentionLayerBase
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)

    for layer_name, attn_module in attn_layers.items():
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
            continue
        if isinstance(attn_module, Attention):
            if spec := attn_module.get_kv_cache_spec(vllm_config):
                kv_cache_spec[layer_name] = spec
                attention_layer_names.append(layer_name)
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
            attention_layer_names.append(layer_name)
            continue

        # Mamba / linear-attention layers are AttentionLayerBase modules but
        # not Attention modules. Keep their native MambaSpec so the v2 runner
        # can create recurrent-state block tables and cache views.
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            if isinstance(spec, MambaSpec):
                # Append Mamba specs after attention specs to preserve the
                # backend ordering expected by Ascend graph parameter updates.
                mamba_specs[layer_name] = spec
            else:
                kv_cache_spec[layer_name] = spec

    if mamba_specs:
        mamba_page_size = max(
            spec.page_size_bytes for spec in mamba_specs.values()
        )
        for layer_name in attention_layer_names:
            spec = kv_cache_spec[layer_name]
            if spec.page_size_bytes < mamba_page_size:
                object.__setattr__(spec, "page_size_padded", mamba_page_size)
        kv_cache_spec.update(mamba_specs)

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
    if seq_lens_cpu_upper_bound is None:
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


def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset) :]


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
    kv_cache_raw_tensors: dict[
        str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ] = {}
    # prefill disaggregation need the addr of cache tensor be aligned with 2M
    alignment = 2 * 1024 * 1024
    layer_kv_cache_spec = _get_layer_kv_cache_specs(kv_cache_config)
    has_mamba = any(
        isinstance(spec, MambaSpec) for spec in layer_kv_cache_spec.values()
    )
    has_attention = any(
        isinstance(spec, AttentionSpec)
        for spec in layer_kv_cache_spec.values()
    )
    use_hybrid_layout = has_mamba and has_attention
    packed_backing: torch.Tensor | None = None
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if len(kv_cache_tensor.shared_by) == 0:
            continue

        # Packed layouts (used when cache groups share one physical backing)
        # must retain a single byte tensor. The per-layer offset and block
        # stride are applied when the logical cache views are created.
        if kv_cache_tensor.block_stride > 0:
            if packed_backing is None:
                packed_size = kv_cache_tensor.size
                if vllm_config.kv_transfer_config is None:
                    packed_backing = torch.zeros(
                        packed_size,
                        dtype=torch.int8,
                        device=device,
                    )
                else:
                    packed_backing = torch.zeros(
                        packed_size + alignment,
                        dtype=torch.int8,
                        device=device,
                    )
                    packed_backing = _align_memory(
                        packed_backing,
                        alignment,
                    )[:packed_size]
            else:
                assert packed_backing.numel() == kv_cache_tensor.size
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = packed_backing
            continue

        # NOTE: We need to init k_cache tensor (nope cache tensor in mla) and
        # v_cache tensor (rope cache tensor in mla) separately to support
        # prefill disaggregation, as it only supports the 0-dim of kv_cache is
        # `num_blocks`.
        # For deepseek mla, we need to spilt cache tensor accrodding to the nope
        # head dim and rope head dim.
        example_layer_name = kv_cache_tensor.shared_by[0]
        example_kv_cache_spec = layer_kv_cache_spec[example_layer_name]

        # Hybrid groups share one raw buffer. Mamba views use the leading
        # conv/SSM regions while attention K/V views overlap the aligned tail,
        # matching the v1 Ascend cache layout.
        contains_mamba = any(
            isinstance(layer_kv_cache_spec[layer_name], MambaSpec)
            for layer_name in kv_cache_tensor.shared_by
        )
        if contains_mamba or use_hybrid_layout:
            tensor_size = kv_cache_tensor.size
            if vllm_config.kv_transfer_config is None:
                tensor = torch.zeros(tensor_size, dtype=torch.int8, device=device)
            else:
                tensor = torch.zeros(
                    tensor_size + alignment,
                    dtype=torch.int8,
                    device=device,
                )
                tensor = _align_memory(tensor, alignment)[:tensor_size]
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor
            continue

        assert isinstance(example_kv_cache_spec, AttentionSpec)

        k_dim, v_dim = _get_attention_kv_cache_dims(example_layer_name, example_kv_cache_spec)
        assert k_dim > 0 and v_dim > 0
        kv_head_dim_list = [k_dim, v_dim]
        if enable_fa_quant(vllm_config):
            k_tensor_split_factor, v_tensor_split_factor = vllm_config.quant_config.get_kv_quant_split_factor(
                example_layer_name, kv_head_dim_list
            )
        else:
            k_tensor_split_factor, v_tensor_split_factor = calc_split_factor(kv_head_dim_list)
        k_tensor_size = int(kv_cache_tensor.size // k_tensor_split_factor)
        v_tensor_size = int(kv_cache_tensor.size // v_tensor_split_factor)

        if vllm_config.kv_transfer_config is None:
            k_tensor = torch.zeros(k_tensor_size, dtype=torch.int8, device=device)
            v_tensor = torch.zeros(v_tensor_size, dtype=torch.int8, device=device)
        else:
            k_tensor = torch.zeros(k_tensor_size + alignment, dtype=torch.int8, device=device)
            v_tensor = torch.zeros(v_tensor_size + alignment, dtype=torch.int8, device=device)
            k_tensor = _align_memory(k_tensor, alignment)[:k_tensor_size]
            v_tensor = _align_memory(v_tensor, alignment)[:v_tensor_size]
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)

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
    kv_cache_raw_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    attn_backends: dict[str, AttentionBackend],
    cache_dtype: str,
    kernel_block_sizes: list[int] | None = None,
    shared_kv_cache_layers: dict[str, str] | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
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
    vllm_config = get_current_vllm_config()

    kv_caches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
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
                raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_k_tensor is not None
                assert raw_v_tensor is not None
                sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
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
                if not isinstance(kv_cache_spec, AscendMLAAttentionSpec):
                    k_shape = kv_cache_shape[1:]
                    if hasattr(kv_cache_spec, "head_size_v"):
                        v_shape = (*kv_cache_shape[1:-1], kv_cache_spec.head_size_v)
                    else:
                        v_shape = k_shape
                else:
                    # k_cache: nope_cache    v_cache: rope_cache
                    mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                    k_dim, v_dim = _get_attention_kv_cache_dims(layer_name, kv_cache_spec)
                    k_shape = (mla_num_blocks, mla_block_size, num_kv_heads, k_dim)
                    v_shape = (mla_num_blocks, mla_block_size, num_kv_heads, v_dim)

                k_cache_dtype = v_cache_dtype = kv_cache_spec.dtype
                if enable_fa_quant(vllm_config):
                    k_cache_dtype, v_cache_dtype = vllm_config.quant_config.get_kv_quant_dtype(
                        layer_name, kv_cache_spec.dtype, vllm_config.model_config
                    )

                k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)
                v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape)
                kv_caches[layer_name] = (k_cache, v_cache)
            else:
                raise ValueError("Unknown KV cache spec type.")

    if shared_kv_cache_layers:
        for layer_name, target_layer_name in shared_kv_cache_layers.items():
            kv_caches[layer_name] = kv_caches[target_layer_name]

    return kv_caches


def _reshape_kv_cache_v2(
    attn_groups: Sequence[AttentionGroup],
    kv_cache_raw_tensors: dict[
        str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ],
    cache_dtype: str,
    kernel_block_sizes: list[int],
    shared_kv_cache_layers: dict[str, str],
    kv_cache_config: "KVCacheConfig | None" = None,
) -> dict[str, Any]:
    vllm_config = get_current_vllm_config()
    is_kv_consumer = (
        vllm_config.kv_transfer_config.is_kv_consumer if vllm_config.kv_transfer_config is not None else False
    )

    layer_packing: dict[str, tuple[int, int]] = {}
    if kv_cache_config is not None:
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            if kv_cache_tensor.block_stride > 0:
                for layer_name in kv_cache_tensor.shared_by:
                    layer_packing[layer_name] = (
                        kv_cache_tensor.offset,
                        kv_cache_tensor.block_stride,
                    )

    kv_caches: dict[str, Any] = {}
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

            raw_cache = kv_cache_raw_tensors[layer_name]
            packing = layer_packing.get(layer_name)

            if isinstance(kv_cache_spec, MambaSpec):
                assert isinstance(raw_cache, torch.Tensor)
                if packing is None:
                    assert raw_cache.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_cache.numel() // kv_cache_spec.page_size_bytes
                else:
                    _, block_stride = packing
                    assert raw_cache.numel() % block_stride == 0
                    num_blocks = raw_cache.numel() // block_stride

                state_tensors: list[torch.Tensor] = []
                state_offset_bytes = 0
                for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                    target_shape = (num_blocks, *shape)
                    state_size_bytes = (
                        torch.empty(shape, device="meta").numel()
                        * get_dtype_size(dtype)
                    )
                    if packing is None:
                        # Preserve the contiguous state layout used by the
                        # Ascend v1 runner: all blocks of one state type are
                        # adjacent, followed by the next state type.
                        start = state_offset_bytes * num_blocks
                        end = start + state_size_bytes * num_blocks
                        state = raw_cache[start:end].view(dtype).view(target_shape)
                    else:
                        layer_offset, block_stride = packing
                        dtype_size = get_dtype_size(dtype)
                        storage_offset_bytes = layer_offset + state_offset_bytes
                        assert storage_offset_bytes % dtype_size == 0
                        assert block_stride % dtype_size == 0
                        strides = list(
                            torch.empty(target_shape, device="meta").stride()
                        )
                        strides[0] = block_stride // dtype_size
                        state = torch.as_strided(
                            raw_cache.view(dtype),
                            size=target_shape,
                            stride=tuple(strides),
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                    state_tensors.append(state)
                    state_offset_bytes += state_size_bytes

                assert state_offset_bytes <= kv_cache_spec.page_size_bytes
                kv_caches[layer_name] = state_tensors
                continue

            assert isinstance(kv_cache_spec, AttentionSpec)

            if packing is None:
                if isinstance(raw_cache, tuple):
                    raw_k_tensor, raw_v_tensor = raw_cache
                    sum_page_size_bytes = (
                        raw_k_tensor.numel() + raw_v_tensor.numel()
                    )
                else:
                    # Attention + Mamba hybrid layout. K/V occupy the tail of
                    # the shared tensor and overlap the aligned Mamba SSM and
                    # padding regions.
                    sum_page_size_bytes = raw_cache.numel()
                assert (
                    sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
                )
                num_blocks = (
                    sum_page_size_bytes // kv_cache_spec.page_size_bytes
                )
            else:
                assert isinstance(raw_cache, torch.Tensor)
                _, block_stride = packing
                assert raw_cache.numel() % block_stride == 0
                num_blocks = raw_cache.numel() // block_stride

            num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
            if packing is not None:
                assert num_blocks_per_kv_block == 1, (
                    "Packed Ascend attention caches require the KV manager "
                    "and kernel block sizes to match."
                )
            kernel_num_blocks = num_blocks * num_blocks_per_kv_block

            kv_cache_shape = group.backend.get_kv_cache_shape(
                kernel_num_blocks,
                kernel_block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype,
            )

            if not isinstance(kv_cache_spec, (AscendMLAAttentionSpec, MLAAttentionSpec)):
                k_shape = kv_cache_shape[1:]
                if hasattr(kv_cache_spec, "head_size_v"):
                    v_shape = (*kv_cache_shape[1:-1], kv_cache_spec.head_size_v)
                else:
                    v_shape = k_shape
            else:
                mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                k_dim, v_dim = _get_attention_kv_cache_dims(layer_name, kv_cache_spec)
                k_shape = (mla_num_blocks, mla_block_size, num_kv_heads, k_dim)
                v_shape = (mla_num_blocks, mla_block_size, num_kv_heads, v_dim)

            k_cache_dtype = v_cache_dtype = kv_cache_spec.dtype
            if is_kv_consumer and enable_fa_quant(vllm_config):
                k_cache_dtype, v_cache_dtype = vllm_config.quant_config.get_kv_quant_dtype(
                    layer_name, kv_cache_spec.dtype, vllm_config.model_config
                )

            if packing is None:
                if isinstance(raw_cache, tuple):
                    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)
                    v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape)
                else:
                    k_size = (
                        torch.empty(k_shape, device="meta").numel()
                        * get_dtype_size(k_cache_dtype)
                    )
                    v_size = (
                        torch.empty(v_shape, device="meta").numel()
                        * get_dtype_size(v_cache_dtype)
                    )
                    kv_start = raw_cache.numel() - k_size - v_size
                    assert kv_start >= 0
                    k_cache = (
                        raw_cache[kv_start : kv_start + k_size]
                        .view(k_cache_dtype)
                        .view(k_shape)
                    )
                    v_cache = (
                        raw_cache[kv_start + k_size :]
                        .view(v_cache_dtype)
                        .view(v_shape)
                    )
            else:
                layer_offset, block_stride = packing
                k_dtype_size = get_dtype_size(k_cache_dtype)
                v_dtype_size = get_dtype_size(v_cache_dtype)
                k_page_size = (
                    torch.empty(k_shape[1:], device="meta").numel()
                    * k_dtype_size
                )
                v_page_size = (
                    torch.empty(v_shape[1:], device="meta").numel()
                    * v_dtype_size
                )
                assert (
                    k_page_size + v_page_size
                    <= kv_cache_spec.page_size_bytes
                )
                assert layer_offset % k_dtype_size == 0
                assert (layer_offset + k_page_size) % v_dtype_size == 0
                assert block_stride % k_dtype_size == 0
                assert block_stride % v_dtype_size == 0

                k_strides = list(
                    torch.empty(k_shape, device="meta").stride()
                )
                v_strides = list(
                    torch.empty(v_shape, device="meta").stride()
                )
                k_strides[0] = block_stride // k_dtype_size
                v_strides[0] = block_stride // v_dtype_size
                k_cache = torch.as_strided(
                    raw_cache.view(k_cache_dtype),
                    size=k_shape,
                    stride=tuple(k_strides),
                    storage_offset=layer_offset // k_dtype_size,
                )
                v_cache = torch.as_strided(
                    raw_cache.view(v_cache_dtype),
                    size=v_shape,
                    stride=tuple(v_strides),
                    storage_offset=(layer_offset + k_page_size)
                    // v_dtype_size,
                )
            kv_caches[layer_name] = (k_cache, v_cache)

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
