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
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import vllm
from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import get_dtype_size, get_kv_cache_torch_dtype
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    get_sfa_qsfa_packed_head_dim,
    is_glm5_next_kpool_cache,
)
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
    get_storage_block_size,
)
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.quantization.utils import enable_fa_quant
from vllm_ascend.utils import (
    calc_split_factor,
    enable_sfa,
    enable_sfa_dcp_replicated_indexer,
    get_kv_cache_tensor_layers,
    vllm_version_is,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """Build Ascend-specific KV cache specs for v2 worker patching."""
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

    kv_cache_spec: dict[str, KVCacheSpec] = {}
    attention_layer_names: list[str] = []
    mamba_specs: dict[str, MambaSpec] = {}
    layer_type = AttentionLayerBase
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)
    sfa_dcp_replicated_indexer_size = (
        vllm_config.parallel_config.decode_context_parallel_size
        if enable_sfa_dcp_replicated_indexer(vllm_config)
        else 1
    )

    if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
        c8_k_cache_dtype = torch.float8_e4m3fn
        c8_k_scale_cache_dtype = torch.float32
    else:
        c8_k_cache_dtype = torch.int8
        c8_k_scale_cache_dtype = torch.float16

    for layer_name, attn_module in attn_layers.items():
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
            continue

        spec = attn_module.get_kv_cache_spec(vllm_config)
        if spec is None:
            continue

        if isinstance(spec, MambaSpec):
            # Keep Mamba groups after attention groups. Ascend graph parameter
            # updates rely on this stable backend ordering.
            mamba_specs[layer_name] = spec
            continue

        if isinstance(attn_module, MLAAttention):
            cache_sparse_sfa_c8 = False
            if getattr(attn_module.impl, "fa_quant_layer", False):
                head_size = attn_module.head_size + attn_module.qk_rope_head_dim
                dtype, cache_dtype_str = attn_module.impl.dtype, None
            elif enable_sfa(vllm_config) and bool(getattr(attn_module.impl, "enable_sparse_sfa_c8", False)):
                cache_sparse_sfa_c8 = True
                head_size = get_sfa_qsfa_packed_head_dim(
                    vllm_config.model_config.hf_text_config.kv_lora_rank,
                    vllm_config.model_config.hf_text_config.qk_rope_head_dim,
                )
                dtype = c8_k_cache_dtype
                cache_dtype_str = vllm_config.cache_config.cache_dtype
            else:
                head_size = spec.head_size
                dtype = spec.dtype
                cache_dtype_str = spec.cache_dtype_str
            spec = AscendMLAAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                cache_dtype_str=cache_dtype_str,
                cache_sparse_sfa_c8=cache_sparse_sfa_c8,
            )
        if isinstance(attn_module, DeepseekV32IndexerCache):
            # GLM-5.3-Flash kpool indexer/tail caches keep their own spec.
            if is_glm5_next_kpool_cache(attn_module):
                kv_cache_spec[layer_name] = spec
                continue
            cache_sparse_li_c8 = get_ascend_config().is_sparse_li_c8_layer(layer_name)
            kv_cache_spec[layer_name] = AscendSFAIndexerCacheSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=vllm_config.model_config.hf_text_config.index_head_dim,
                dtype=c8_k_cache_dtype
                if cache_sparse_li_c8
                else get_kv_cache_torch_dtype(
                    vllm_config.cache_config.cache_dtype,
                    vllm_config.model_config.dtype,
                ),
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
                scale_dim=1 if cache_sparse_li_c8 else 0,
                scale_dtype=c8_k_scale_cache_dtype if cache_sparse_li_c8 else torch.int8,
                cache_sparse_li_c8=cache_sparse_li_c8,
                sfa_dcp_replicated_indexer_size=sfa_dcp_replicated_indexer_size,
            )
            continue

        kv_cache_spec[layer_name] = spec
        if isinstance(spec, AttentionSpec):
            attention_layer_names.append(layer_name)
            continue

    if mamba_specs:
        common_page_size = max(spec.page_size_bytes for spec in (*kv_cache_spec.values(), *mamba_specs.values()))
        for layer_name in attention_layer_names:
            spec = kv_cache_spec[layer_name]
            page_size_padded = common_page_size if spec.page_size_bytes < common_page_size else spec.page_size_padded
            # Ascend exposes K and V as separate block-first views even when
            # the backend's logical cache shape starts with the K/V dimension.
            # Consequently, padded pages are indexed by their runtime block
            # stride and are safe for hybrid Attention/Mamba allocations.
            # vLLM #51718 removed AttentionSpec.indexes_kv_by_block_stride on
            # main; page_size_padded alone carries the padding there.
            if vllm_version_is("0.27.1"):
                kv_cache_spec[layer_name] = replace(
                    spec,
                    page_size_padded=page_size_padded,
                    indexes_kv_by_block_stride=True,
                )
            else:
                kv_cache_spec[layer_name] = replace(spec, page_size_padded=page_size_padded)
        for layer_name, spec in mamba_specs.items():
            if spec.page_size_bytes < common_page_size:
                mamba_specs[layer_name] = replace(spec, page_size_padded=common_page_size)
        kv_cache_spec.update(mamba_specs)

    return kv_cache_spec


def build_attn_metadata(
    *,
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_actual_reqs: int | None = None,
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
    num_actual_tokens: int | None = None,
    num_input_tokens: int | None = None,
    is_prefilling: torch.Tensor | None = None,
    pcp_context: "AscendPCPAttentionContext | None" = None,
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

    # Upstream speculative-decoding callers do not provide Ascend's separate
    # scheduled-token and padded-input-token counts. Without these fields,
    # ``num_tokens`` is the only available count and correctly serves as both
    # the actual token count and the model input token count.
    if num_actual_tokens is None:
        num_actual_tokens = num_tokens
    if num_input_tokens is None:
        num_input_tokens = num_tokens
    if num_actual_reqs is None:
        num_actual_reqs = num_reqs

    attn_metadata: dict[str, Any] = {}
    # Share request-level DSA metadata across cache groups in one execution.
    common_ratio_to_sas_metadata: dict[Any, Any] = {}
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
        common_is_prefilling = common_attn_metadata_extra_kwargs.pop(
            "is_prefilling",
            is_prefilling,
        )
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            seq_lens=seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            positions=positions,
            attn_state=attn_state,
            graph_pad_size=graph_pad_size,
            num_input_tokens=num_input_tokens,
            is_prefilling=common_is_prefilling,
            max_seq_len=max_seq_len,
            causal=group_causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
            **common_attn_metadata_extra_kwargs,
        )

        for attn_group in attn_groups[i]:
            attn_metadata_builder = attn_group.get_metadata_builder(0)
            is_dsa_builder = isinstance(attn_metadata_builder, AscendDSAMetadataBuilder)
            attn_metadata_extra_kwargs = (
                model_specific_attn_metadata.get_extra_attn_kwargs(
                    attn_metadata_builder,
                    num_reqs,
                )
                if not for_cudagraph_capture and model_specific_attn_metadata is not None
                else {}
            )
            if is_dsa_builder:
                # DSA cache groups share request-level metadata during replay.
                attn_metadata_extra_kwargs.update(
                    num_actual_reqs=num_actual_reqs,
                    common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                )
                if pcp_context is not None:
                    attn_metadata_extra_kwargs.update(
                        pcp_context=pcp_context,
                        pcp_cache_group_idx=i,
                    )

            if for_cudagraph_capture:
                metadata = attn_metadata_builder.build_for_cudagraph_capture(
                    common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )
            else:
                metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )
            if is_dsa_builder:
                # Preserve sharing even if a builder replaces one of the
                # dictionaries while constructing its metadata.
                common_ratio_to_sas_metadata = attn_metadata_builder.common_ratio_to_sas_metadata  # type: ignore[assignment]
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


def _is_dsv4_model(vllm_config: VllmConfig) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None) if model_config else None
    return hf_config is not None and hasattr(hf_config, "compress_ratios")


def _get_attention_kv_cache_dims(
    layer_name: str,
    kv_cache_spec: AttentionSpec,
) -> tuple[int, int]:
    if isinstance(kv_cache_spec, AscendMLAAttentionSpec):
        attn_layers = get_layers_from_vllm_config(get_current_vllm_config(), AttentionLayerBase, [layer_name])
        attn_layer = attn_layers[layer_name]
        if not isinstance(attn_layer, MLAAttention):
            raise TypeError(f"Expected an MLAAttention layer for {layer_name}, got {type(attn_layer).__name__}.")
        return attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim

    head_size_v = getattr(kv_cache_spec, "head_size_v", kv_cache_spec.head_size)
    return kv_cache_spec.head_size, head_size_v


def _adjust_dsv4_kv_layout(
    raw_tensor: torch.Tensor,
    cache_shapes: list[tuple[int, ...]],
    cache_dtypes: list[torch.dtype],
    page_size_bytes: int,
    overlap_full_kv_cache: bool = False,
) -> list[torch.Tensor]:
    caches = []
    base_offset_bytes = raw_tensor.storage_offset() * raw_tensor.element_size()
    offset_bytes = base_offset_bytes
    for index, (shape, dtype) in enumerate(zip(cache_shapes, cache_dtypes)):
        if overlap_full_kv_cache and index == 2:
            offset_bytes = base_offset_bytes
        dtype_size = get_dtype_size(dtype)
        page_stride = page_size_bytes // dtype_size
        stride = torch.empty(shape).stride()
        if offset_bytes % dtype_size:
            raise ValueError(f"DSA cache offset {offset_bytes} is not aligned to {dtype}.")
        caches.append(
            torch.as_strided(
                raw_tensor.view(dtype),
                size=shape,
                stride=(page_stride, *stride[1:]),
                storage_offset=offset_bytes // dtype_size,
            )
        )
        offset_bytes += stride[0] * dtype_size
    return caches


def _view_dsv4_cache(
    raw_tensor: torch.Tensor,
    kv_cache_spec: AttentionSpec,
    attn_backend: AttentionBackend,
    kv_cache_config: KVCacheConfig,
) -> list[torch.Tensor]:
    """Create DSA cache views without applying normal MLA K/V splitting."""
    if raw_tensor.numel() % kv_cache_spec.page_size_bytes:
        raise ValueError("DSA cache allocation is not a whole number of physical pages.")
    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
    if num_blocks != kv_cache_config.num_blocks:
        raise ValueError(f"DSA cache has {num_blocks} blocks, expected {kv_cache_config.num_blocks}.")

    k_shape = attn_backend.get_kv_cache_shape(
        num_blocks,
        get_storage_block_size(kv_cache_spec),
        kv_cache_spec.num_kv_heads,
        kv_cache_spec.head_size,
    )
    cache_shapes = [k_shape]
    cache_dtypes = [kv_cache_spec.dtype]
    overlap_full_kv_cache = False

    scale_dim = int(getattr(kv_cache_spec, "scale_dim", 0))
    if scale_dim:
        scale_dtype = kv_cache_spec.scale_dtype
        scale_shape = attn_backend.get_kv_cache_shape(
            num_blocks,
            get_storage_block_size(kv_cache_spec),
            kv_cache_spec.num_kv_heads,
            scale_dim,
        )
        cache_shapes.append(scale_shape)
        cache_dtypes.append(scale_dtype)
        if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
            full_shape = attn_backend.get_kv_cache_shape(
                num_blocks,
                get_storage_block_size(kv_cache_spec),
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size + scale_dim * get_dtype_size(scale_dtype),
            )
            cache_shapes.append(full_shape)
            cache_dtypes.append(kv_cache_spec.dtype)
            overlap_full_kv_cache = True

    return _adjust_dsv4_kv_layout(
        raw_tensor,
        cache_shapes,
        cache_dtypes,
        kv_cache_spec.page_size_bytes,
        overlap_full_kv_cache,
    )


def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset) :]


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _allocate_int8_cache_tensor(
    numel: int,
    alignment: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate an int8 raw cache tensor.

    When KV transfer is enabled, the returned tensor's data_ptr is aligned
    to `alignment`. This keeps the original Mooncake/ADXL alignment behavior.
    """
    if numel <= 0:
        raise ValueError(f"Invalid cache tensor size: {numel}")

    vllm_config = get_current_vllm_config()
    if vllm_config.kv_transfer_config is None:
        return torch.zeros(numel, dtype=torch.int8, device=device)

    raw_tensor = torch.zeros(
        numel + alignment,
        dtype=torch.int8,
        device=device,
    )
    return _align_memory(raw_tensor, alignment)[:numel]


def _allocate_sparse_c8_indexer_tensors(
    dsa_k_tensor_size: int,
    dsa_k_scale_tensor_size: int,
    alignment: int,
    scale_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate dsa_k and dsa_k_scale from one aligned int8 raw allocation.

    Both returned tensors are logical views into the same underlying storage:

        sparse_c8_raw
            ├── dsa_k_tensor        int8 raw bytes
            └── dsa_k_scale_tensor  scale dtype raw bytes stored as int8 view

    `dsa_k_scale_tensor` is still returned as int8 raw storage. Later reshape
    code should continue to use:

        raw_dsa_k_scale_tensor.view(scale_dtype).view(scale_shape)

    This reduces HCCL/Mooncake registration count because register_buffer
    can merge these two views into one registered memory range.
    """
    if dsa_k_tensor_size <= 0:
        raise ValueError(f"Invalid dsa_k_tensor_size: {dsa_k_tensor_size}")
    if dsa_k_scale_tensor_size <= 0:
        raise ValueError(f"Invalid dsa_k_scale_tensor_size: {dsa_k_scale_tensor_size}")

    scale_dtype_size = torch.empty((), dtype=scale_dtype).element_size()

    # Ensure the scale view starts at an address aligned for scale_dtype.
    scale_offset = _align_up(dsa_k_tensor_size, scale_dtype_size)
    total_raw_size = scale_offset + dsa_k_scale_tensor_size

    sparse_c8_raw_tensor = _allocate_int8_cache_tensor(
        total_raw_size,
        alignment,
        device,
    )

    dsa_k_tensor = sparse_c8_raw_tensor[:dsa_k_tensor_size]
    dsa_k_scale_tensor = sparse_c8_raw_tensor[scale_offset : scale_offset + dsa_k_scale_tensor_size]

    assert dsa_k_tensor.is_contiguous()
    assert dsa_k_scale_tensor.is_contiguous()
    assert dsa_k_scale_tensor.data_ptr() % scale_dtype_size == 0
    assert dsa_k_scale_tensor.numel() % scale_dtype_size == 0

    return dsa_k_tensor, dsa_k_scale_tensor


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
    is_dsv4_model = _is_dsv4_model(vllm_config)
    # init kv cache tensors
    kv_cache_raw_tensors: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
    # prefill disaggregation need the addr of cache tensor be aligned with 2M
    alignment = 2 * 1024 * 1024
    layer_kv_cache_spec = _get_layer_kv_cache_specs(kv_cache_config)
    has_mamba = any(isinstance(spec, MambaSpec) for spec in layer_kv_cache_spec.values())
    has_attention = any(isinstance(spec, AttentionSpec) for spec in layer_kv_cache_spec.values())
    use_hybrid_layout = has_mamba and has_attention

    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        shared_names = get_kv_cache_tensor_layers(kv_cache_tensor)
        if not shared_names:
            continue

        if is_dsv4_model:
            # DSA reshapes it with its own page-strided layout below.
            if vllm_config.kv_transfer_config is None:
                raw_tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
            else:
                raw_tensor = torch.zeros(
                    kv_cache_tensor.size + alignment,
                    dtype=torch.int8,
                    device=device,
                )
                raw_tensor = _align_memory(raw_tensor, alignment)[: kv_cache_tensor.size]
            for layer_name in shared_names:
                kv_cache_raw_tensors[layer_name] = raw_tensor
            continue

        example_layer_name = shared_names[0]
        example_spec = layer_kv_cache_spec[example_layer_name]

        # Use one raw allocation for Mamba and hybrid caches. The reshape step
        # creates the V1-compatible contiguous state views and overlaps
        # Attention K/V with the aligned tail of the same buffer.
        contains_mamba = any(isinstance(layer_kv_cache_spec[layer_name], MambaSpec) for layer_name in shared_names)
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
            for layer_name in shared_names:
                kv_cache_raw_tensors[layer_name] = tensor
            continue
        assert isinstance(example_spec, AttentionSpec)

        if isinstance(example_spec, AscendSFAIndexerCacheSpec):
            num_blocks = kv_cache_tensor.size // example_spec.page_size_bytes
            if not vllm_version_is("0.27.1"):
                # vLLM #51718 packs all group layers into one tensor;
                # kv_cache_config.num_blocks is the per-layer block count.
                num_blocks = kv_cache_config.num_blocks

            k_tensor_size = (
                num_blocks
                * example_spec.sfa_dcp_replicated_indexer_size
                * example_spec.block_size
                * example_spec.num_kv_heads
                * example_spec.head_size
                * get_dtype_size(example_spec.dtype)
            )
            if example_spec.scale_dim:
                scale_tensor_size = (
                    num_blocks
                    * example_spec.sfa_dcp_replicated_indexer_size
                    * example_spec.block_size
                    * example_spec.num_kv_heads
                    * example_spec.scale_dim
                    * get_dtype_size(example_spec.scale_dtype)
                )
            else:
                scale_tensor_size = None

            if vllm_version_is("0.27.1"):
                # v0.27.1 `shared_by` aliases the same physical blocks.
                if scale_tensor_size is not None:
                    kv_cache_raw_tensors[shared_names[0]] = _allocate_sparse_c8_indexer_tensors(
                        dsa_k_tensor_size=k_tensor_size,
                        dsa_k_scale_tensor_size=scale_tensor_size,
                        alignment=alignment,
                        scale_dtype=example_spec.scale_dtype,
                        device=device,
                    )
                else:
                    kv_cache_raw_tensors[shared_names[0]] = (
                        _allocate_int8_cache_tensor(k_tensor_size, alignment, device),
                    )
                for layer_name_inner in shared_names[1:]:
                    kv_cache_raw_tensors[layer_name_inner] = kv_cache_raw_tensors[shared_names[0]]
            else:
                # main: every layer owns its own region.
                for layer_name_inner in shared_names:
                    if scale_tensor_size is not None:
                        kv_cache_raw_tensors[layer_name_inner] = _allocate_sparse_c8_indexer_tensors(
                            dsa_k_tensor_size=k_tensor_size,
                            dsa_k_scale_tensor_size=scale_tensor_size,
                            alignment=alignment,
                            scale_dtype=example_spec.scale_dtype,
                            device=device,
                        )
                    else:
                        kv_cache_raw_tensors[layer_name_inner] = (
                            _allocate_int8_cache_tensor(k_tensor_size, alignment, device),
                        )

            continue

        # vLLM #51718 packs all group layers into one tensor on main; each layer
        # owns a page_size * num_blocks region (the per-layer size v0.27.1 stored
        # directly on the tensor).
        kv_cache_tensor_size = (
            kv_cache_tensor.size
            if vllm_version_is("0.27.1")
            else kv_cache_config.num_blocks * example_spec.page_size_bytes
        )
        # TODO:Subsequently, extend the `AttentionSpec` class in the vLLM community and remove these branches.
        if enable_sfa(vllm_config) and bool(getattr(example_spec, "cache_sparse_sfa_c8", False)):
            k_size = kv_cache_tensor_size
            if vllm_version_is("0.27.1"):
                k_tensor = _allocate_int8_cache_tensor(k_size, alignment, device)
                for layer_name in shared_names:
                    kv_cache_raw_tensors[layer_name] = k_tensor
            else:
                for layer_name in shared_names:
                    kv_cache_raw_tensors[layer_name] = _allocate_int8_cache_tensor(k_size, alignment, device)
        else:
            k_dim, v_dim = _get_attention_kv_cache_dims(example_layer_name, example_spec)
            if enable_fa_quant(vllm_config):
                k_factor, v_factor = vllm_config.quant_config.get_kv_quant_split_factor(
                    example_layer_name, [k_dim, v_dim]
                )
            else:
                k_factor, v_factor = calc_split_factor([k_dim, v_dim])
            k_size = int(kv_cache_tensor_size // k_factor)
            v_size = int(kv_cache_tensor_size // v_factor)
            if vllm_version_is("0.27.1"):
                k_tensor = _allocate_int8_cache_tensor(k_size, alignment, device)
                v_tensor = _allocate_int8_cache_tensor(v_size, alignment, device)
                for layer_name in shared_names:
                    kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)
            else:
                for layer_name in shared_names:
                    k_tensor = _allocate_int8_cache_tensor(k_size, alignment, device)
                    v_tensor = _allocate_int8_cache_tensor(v_size, alignment, device)
                    kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)

    layer_names = {layer_name for group in kv_cache_config.kv_cache_groups for layer_name in group.layer_names}
    assert layer_names == (kv_cache_raw_tensors.keys() | shared_layers.keys()), (
        "Some layers are not correctly initialized"
    )
    return kv_cache_raw_tensors


def _reshape_mamba_kv_cache(
    raw_cache: torch.Tensor,
    kv_cache_spec: MambaSpec,
) -> list[torch.Tensor]:
    """Create the contiguous per-state views used by the Ascend v1 runner."""
    page_size_bytes = kv_cache_spec.page_size_bytes
    assert raw_cache.numel() % page_size_bytes == 0
    num_blocks = raw_cache.numel() // page_size_bytes

    state_tensors: list[torch.Tensor] = []
    start_idx = 0
    # Keep the same hybrid storage layout as model_runner_v1:
    #
    # tensor1: [(kv_padding), conv, ...]
    # tensor2: [k,            ssm,  ...]
    # tensor3: [v,            (mamba_padding), ...]
    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
        target_shape = (num_blocks, *shape)
        end_idx = start_idx + torch.empty(
            target_shape,
            device="meta",
        ).numel() * get_dtype_size(dtype)
        state = raw_cache[start_idx:end_idx].view(dtype).view(target_shape)
        state_tensors.append(state)
        start_idx = end_idx

    assert start_idx <= raw_cache.numel()
    return state_tensors


def _reshape_kv_cache_v2(
    attn_groups: Sequence[AttentionGroup],
    kv_cache_raw_tensors: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    cache_dtype: str,
    kernel_block_sizes: list[int],
    shared_kv_cache_layers: dict[str, str],
    kv_cache_config: "KVCacheConfig | None" = None,
) -> dict[str, Any]:
    if kv_cache_config is None:
        raise ValueError("Reshape KV cache requires KVCacheConfig.")

    vllm_config = get_current_vllm_config()
    is_dsv4_model = _is_dsv4_model(vllm_config)
    layer_kv_cache_spec = _get_layer_kv_cache_specs(kv_cache_config)
    kv_caches: dict[str, Any] = {}

    for group in attn_groups:
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue

        group_spec = group.kv_cache_spec
        group_storage_block_size = get_storage_block_size(group_spec)
        kernel_block_size = (
            group_storage_block_size
            if group_storage_block_size != group_spec.block_size
            else kernel_block_sizes[group.kv_cache_group_id]
        )

        for layer_name in group.layer_names:
            if layer_name in shared_kv_cache_layers:
                continue

            kv_cache_spec = layer_kv_cache_spec[layer_name]

            if isinstance(group_spec, AscendSFAIndexerCacheSpec):
                assert kv_cache_config is not None
                raw_cache = kv_cache_raw_tensors[layer_name]
                assert isinstance(raw_cache, tuple)

                if group_spec.scale_dim:
                    raw_k_tensor, raw_scale_tensor = raw_cache
                    sum_page_size_bytes = raw_k_tensor.numel() + raw_scale_tensor.numel()
                else:
                    (raw_k_tensor,) = raw_cache
                    raw_scale_tensor = None
                    sum_page_size_bytes = raw_k_tensor.numel()

                assert sum_page_size_bytes % group_spec.page_size_bytes == 0
                num_blocks = sum_page_size_bytes // group_spec.page_size_bytes
                assert num_blocks >= kv_cache_config.num_blocks

                kv_cache_shape = group.backend.get_kv_cache_shape(
                    num_blocks * group_spec.sfa_dcp_replicated_indexer_size,
                    group_spec.block_size,
                    group_spec.num_kv_heads,
                    group_spec.head_size,
                )

                indexer_k_cache = raw_k_tensor.view(group_spec.dtype).view(kv_cache_shape)
                if raw_scale_tensor is None:
                    kv_caches[layer_name] = (indexer_k_cache,)
                else:
                    indexer_scale_cache_shape = group.backend.get_kv_cache_shape(
                        num_blocks * group_spec.sfa_dcp_replicated_indexer_size,
                        group_spec.block_size,
                        group_spec.num_kv_heads,
                        group_spec.scale_dim,
                    )
                    indexer_scale_cache = raw_scale_tensor.view(group_spec.scale_dtype).view(indexer_scale_cache_shape)
                    kv_caches[layer_name] = (indexer_k_cache, indexer_scale_cache)

                continue

            raw_cache = kv_cache_raw_tensors[layer_name]
            if is_dsv4_model and isinstance(
                kv_cache_spec,
                (AscendMLAAttentionSpec, AscendSlidingWindowMLASpec),
            ):
                if not isinstance(raw_cache, torch.Tensor):
                    raise ValueError(f"DSA cache for {layer_name} must use one raw tensor.")
                kv_caches[layer_name] = _view_dsv4_cache(
                    raw_cache,
                    kv_cache_spec,
                    group.backend,
                    kv_cache_config,
                )
                continue

            if isinstance(kv_cache_spec, MambaSpec):
                if not isinstance(raw_cache, torch.Tensor):
                    raise ValueError(f"Mamba cache for {layer_name} must use one raw tensor.")
                mamba_cache = _reshape_mamba_kv_cache(raw_cache, kv_cache_spec)
                if mamba_cache[0].shape[0] < kv_cache_config.num_blocks:
                    raise ValueError(f"Mamba cache for {layer_name} has fewer blocks than KVCacheManager.")
                kv_caches[layer_name] = mamba_cache
                continue

            if not isinstance(kv_cache_spec, AttentionSpec):
                raise TypeError(f"Unsupported KV cache spec: {type(kv_cache_spec).__name__}.")

            if isinstance(raw_cache, tuple):
                raw_k_tensor, raw_v_tensor = raw_cache
                total_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
            else:
                # Attention and Mamba share one aligned raw allocation.
                total_bytes = raw_cache.numel()

            if total_bytes % kv_cache_spec.page_size_bytes:
                raise ValueError(f"KV cache for {layer_name} is not a whole number of pages.")
            num_blocks = total_bytes // kv_cache_spec.page_size_bytes
            num_blocks_per_kv_block = get_storage_block_size(kv_cache_spec) // kernel_block_size
            kernel_num_blocks = num_blocks * num_blocks_per_kv_block
            kv_cache_shape = group.backend.get_kv_cache_shape(
                kernel_num_blocks,
                kernel_block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype,
            )
            sparse_sfa_c8 = enable_sfa(vllm_config) and bool(getattr(kv_cache_spec, "cache_sparse_sfa_c8", False))
            if isinstance(kv_cache_spec, (AscendMLAAttentionSpec, MLAAttentionSpec)):
                num_blocks_, block_size_, num_kv_heads, _ = kv_cache_shape
                k_dim, v_dim = _get_attention_kv_cache_dims(layer_name, kv_cache_spec)
                k_shape = (num_blocks_, block_size_, num_kv_heads, k_dim)
                if sparse_sfa_c8:
                    k_shape = (num_blocks_, block_size_, num_kv_heads, kv_cache_spec.head_size)
                    v_dim = 0
                v_shape = (num_blocks_, block_size_, num_kv_heads, v_dim)
            else:
                k_shape = kv_cache_shape[1:]
                v_shape = (
                    *kv_cache_shape[1:-1],
                    getattr(kv_cache_spec, "head_size_v", kv_cache_spec.head_size),
                )

            k_dtype = v_dtype = kv_cache_spec.dtype
            if enable_fa_quant(vllm_config):
                k_dtype, v_dtype = vllm_config.quant_config.get_kv_quant_dtype(
                    layer_name,
                    kv_cache_spec.dtype,
                    vllm_config.model_config,
                )

            if sparse_sfa_c8:
                raw_k_tensor = raw_cache
                k_dtype = (
                    torch.float8_e4m3fn
                    if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)
                    else torch.int8
                )
                k_cache = raw_k_tensor.view(k_dtype).view(k_shape)
                kv_caches[layer_name] = (k_cache,)
            elif isinstance(raw_cache, tuple):
                raw_k_tensor, raw_v_tensor = raw_cache
                k_cache = raw_k_tensor.view(k_dtype).view(k_shape)
                v_cache = raw_v_tensor.view(v_dtype).view(v_shape)
                kv_caches[layer_name] = (k_cache, v_cache)
            else:
                # Keep Attention K/V contiguous across the tail of the hybrid
                # allocation, matching the model_runner_v1 storage contract.
                k_size = torch.empty(k_shape, device="meta").numel() * get_dtype_size(k_dtype)
                v_size = torch.empty(v_shape, device="meta").numel() * get_dtype_size(v_dtype)
                kv_start = raw_cache.numel() - k_size - v_size
                if kv_start < 0:
                    raise ValueError(f"Attention cache views exceed the allocation for {layer_name}.")
                k_cache = raw_cache[kv_start : kv_start + k_size].view(k_dtype).view(k_shape)
                v_cache = raw_cache[kv_start + k_size :].view(v_dtype).view(v_shape)
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


@contextmanager
def build_draft_attn_metadata_factory(positions, pad, is_prefilling):
    """Wrap build_attn_metadata to forward rotary positions for the draft block.

    The generic (Ascend) ``build_attn_metadata`` reads ``positions`` inside the
    DSA/MLA ``build_decode_metadata`` for cos/sin, but the flat upstream
    speculator path does not forward them. Must run inside
    ``build_attn_metadata_wrapper()``.
    """
    raw = _BUILD_ATTN_METADATA_MODULE.build_attn_metadata  # cache

    def build_attn_metadata(*args, **kwargs):
        kwargs["positions"] = positions[:pad]
        kwargs["is_prefilling"] = is_prefilling
        return raw(*args, **kwargs)

    try:
        _BUILD_ATTN_METADATA_MODULE.build_attn_metadata = build_attn_metadata
        yield
    finally:
        _BUILD_ATTN_METADATA_MODULE.build_attn_metadata = raw  # restore
