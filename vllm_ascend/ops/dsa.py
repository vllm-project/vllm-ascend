# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from dataclasses import dataclass

import torch
import torch_npu
from torch import nn
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

from vllm_ascend.attention.dsa_v1 import dsv4_dsa_overlap_stream
from vllm_ascend.models.layer.attention.layer import DSAAttention
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
    npu_stream_switch,
)


@dataclass
class DSAModules:
    """Modules used in SFA V2."""

    wq_a: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    wkv: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    indexer: torch.nn.Module | None
    compressor: torch.nn.Module | None
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None
    skip_topk: bool = False


class AscendDeepseekSparseAttention(MultiHeadLatentAttentionWrapper):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        eps: float,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        dsa_modules: DSAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.dim = dim
        self.n_heads = n_heads
        self.scale = scale
        self.n_local_heads = n_local_heads
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = nope_head_dim
        self.eps = eps
        self.n_groups = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.compress_ratio = compress_ratio

        self.wq_a = dsa_modules.wq_a
        self.q_norm = dsa_modules.q_norm
        self.wq_b = dsa_modules.wq_b
        self.wkv = dsa_modules.wkv
        self.kv_norm = dsa_modules.kv_norm
        self.wo_a = dsa_modules.wo_a
        self.wo_b = dsa_modules.wo_b
        self.attn_sink = dsa_modules.attn_sink
        self.indexer = dsa_modules.indexer
        self.compressor = dsa_modules.compressor
        self.topk_indices_buffer = dsa_modules.topk_indices_buffer
        self.indexer_rotary_emb = dsa_modules.indexer_rotary_emb
        self.skip_topk = dsa_modules.skip_topk
        self.prefix = prefix

        ascend_device_type = get_ascend_device_type()
        k_dtype = torch.fp8 if ascend_device_type == AscendDeviceType.A5 else torch.bfloat16
        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=k_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        self.dsa_attn = DSAAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            # extra
            wq_a=self.wq_a,
            wq_b=self.wq_b,
            wkv=self.wkv,
            q_norm=self.q_norm,
            kv_norm=self.kv_norm,
            indexer=self.indexer,
            compressor=self.compressor,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            eps=self.eps,
            swa_cache_layer=self.swa_cache_layer,
            skip_topk=self.skip_topk,
            topk_indices_buffer=self.topk_indices_buffer,
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        need_gather_q_kv = get_forward_context().flash_comm_v1_enabled
        output_shape = hidden_states.shape

        output = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)

        # All DSA forward paths run inside dsa_forward custom op boundary,
        # which is required for ACL graph capture (registered with
        # dispatch_key="PrivateUse1").  When dual-stream is disabled,
        # dsa_forward dispatches to the original serial path.
        torch.ops.vllm.dsa_forward(hidden_states, need_gather_q_kv, output, self.prefix)

        output = output.view(-1, output_shape[-1])
        return output


def dsa_forward(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if forward_context.attn_metadata:
        attn_metadata = filter_metadata(forward_context.attn_metadata, self.prefix)
    else:
        attn_metadata = forward_context.attn_metadata

    if attn_metadata is None:
        # Profiling run.
        # When dual-stream is enabled, the aux stream runs ops during forward that have never been
        # exercised during profiling. This warmup ensures all aux-stream op patterns are captured
        # for ACL graph compatibility.
        impl = self.dsa_attn.impl
        if hasattr(impl, "multistream_dsv4_dsa_overlap") and impl.multistream_dsv4_dsa_overlap:
            dummy = torch.zeros(1, hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device)
            aux_stream = dsv4_dsa_overlap_stream()
            e_warmup = torch.npu.current_stream().record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_warmup)
                if hasattr(impl.wkv, "weight_scale") and impl.wkv.weight.dtype == torch.int8:
                    kv_q_dummy, kv_s_dummy = torch_npu.npu_dynamic_quant(dummy)
                    _ = torch_npu.npu_quant_matmul(
                        kv_q_dummy,
                        impl.wkv.weight,
                        impl.wkv.weight_scale,
                        pertoken_scale=kv_s_dummy,
                        output_dtype=hidden_states.dtype,
                    )
                else:
                    _ = impl.cv_wkv.quantize(dummy)
                    _ = impl.cv_wkv.matmul(dummy, None)
                kv_dummy = torch.zeros(
                    1, impl.nope_head_dim + impl.rope_head_dim, dtype=hidden_states.dtype, device=hidden_states.device
                )
                _ = impl.kv_norm(kv_dummy)

                # indexer module aux stream ops
                # Part1 aux: kv_quant (npu_dynamic_quant)
                soc_version = get_ascend_device_type()
                dst_type = torch.float8_e4m3fn if soc_version == AscendDeviceType.A5 else torch.int8
                kv_dummy, kv_scale_dummy = torch_npu.npu_dynamic_quant(dummy, dst_type=dst_type)
                # Part1 aux: scatter_k_cache (npu_scatter_nd_update_v2)
                # In profiling stage, create dummy tensors to ensure ACL graph captures scatter operator.
                if self.compress_ratio == 4 and self.indexer is not None:
                    slot_mapping_dummy = torch.zeros(1, dtype=torch.int64, device=hidden_states.device)
                    # Create dummy tensors for scatter warmup
                    dummy_shape = (1, 1, 1, kv_dummy.shape[-1])  # [num_blocks, block_size, num_heads, head_dim]
                    indexer_k_cache = torch.zeros(dummy_shape, dtype=kv_dummy.dtype, device=hidden_states.device)
                    indexer_scale_cache = torch.zeros(dummy_shape, dtype=torch.float16, device=hidden_states.device)

                    torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_k_cache, slot_mapping_dummy, kv_dummy)
                    # Part3 aux: scatter_scale_cache (npu_scatter_nd_update_v2)
                    kv_scale_dummy = kv_scale_dummy.to(torch.float16).unsqueeze(-1)
                    torch.ops._C_ascend.npu_scatter_nd_update_v2(
                        indexer_scale_cache, slot_mapping_dummy, kv_scale_dummy
                    )

                    # Part4 kv_comprecessor module
                    _ = impl.weights_proj(dummy)

            torch.npu.current_stream().wait_stream(aux_stream)
        output.fill_(0)
        return

    kv_cache = _build_kv_cache(self, forward_context)

    impl = self.dsa_attn.impl
    has_decode = attn_metadata[0].num_decodes > 0
    has_prefill = attn_metadata[0].num_prefills > 0
    use_dual = impl._use_dual_stream()

    if use_dual and self.compress_ratio == 4 and (has_decode or has_prefill):
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, need_gather_q_kv)

        decode_tokens = attn_metadata[0].num_decode_tokens
        actual_tokens = attn_metadata[0].num_actual_tokens

        # ============================================================
        # Phase 1: Q/KV compute + indexer prepare (per phase)
        # ============================================================
        decode_result = None
        prefill_result = None

        if has_decode:
            decode_hs = hidden_states[:decode_tokens]
            decode_result = impl.dsa_decode_prepare(self.dsa_attn.layer_name, decode_hs, kv_cache, attn_metadata)

        if has_prefill:
            prefill_hs = hidden_states[decode_tokens:actual_tokens]
            prefill_result = impl.dsa_prefill_prepare(self.dsa_attn.layer_name, prefill_hs, kv_cache, attn_metadata)

        # ============================================================
        # Phase 2: Compressor + dual-stream weights_proj overlap
        # ============================================================
        (compressor_attn_metadata, compressor_kv_state_metadata, _, _, _) = attn_metadata

        coff = 2 if impl.compressor_overlap else 1
        unfolded_state_cache = kv_cache[2]

        # Decode compressor
        decode_compressed_kv = None
        if has_decode:
            assert decode_result is not None
            (q, compress_cos, compress_sin, actual_seq_lengths_query, q_idx, kv_idx, ik, isc, isc_meta, wp) = (
                decode_result
            )

            decode_compressed_kv = torch.ops._C_ascend.compressor(
                decode_hs,
                impl.compressor_wkv.weight,
                impl.compressor_wgate.weight,
                unfolded_state_cache.squeeze(-2),
                impl.compressor_ape,
                impl.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                state_block_table=compressor_kv_state_metadata.decode.block_table,
                cu_seqlens=actual_seq_lengths_query,
                seqused=None,
                start_pos=compressor_attn_metadata.decode.start_pos,
                rope_head_dim=impl.rope_head_dim,
                cmp_ratio=impl.compress_ratio,
                coff=coff,
                norm_eps=impl.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )

        # Prefill compressor
        prefill_compressed_kv = None
        if has_prefill:
            assert prefill_result is not None
            (
                pq,
                pcompress_cos,
                pcompress_sin,
                pactual_seq_lengths_query,
                pq_idx,
                pkv_idx,
                pik,
                pisc,
                pisc_meta,
                pwp,
            ) = prefill_result

            prefill_compressed_kv = torch.ops._C_ascend.compressor(
                prefill_hs,
                impl.compressor_wkv.weight,
                impl.compressor_wgate.weight,
                unfolded_state_cache.squeeze(-2),
                impl.compressor_ape,
                impl.compressor_norm.weight,
                pcompress_sin.view(-1, pcompress_sin.shape[-1]),
                pcompress_cos.view(-1, pcompress_cos.shape[-1]),
                state_block_table=compressor_kv_state_metadata.prefill.block_table,
                cu_seqlens=pactual_seq_lengths_query,
                seqused=None,
                start_pos=compressor_attn_metadata.prefill.start_pos,
                rope_head_dim=impl.rope_head_dim,
                cmp_ratio=impl.compress_ratio,
                coff=coff,
                norm_eps=impl.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )
            if prefill_compressed_kv.numel() == 0:
                prefill_compressed_kv = None

        # Dual-stream: weights_proj on sub-stream overlaps with
        # quant_scatter + scatter on main stream
        e1 = torch.npu.current_stream().record_event()

        aux_stream = dsv4_dsa_overlap_stream()
        with npu_stream_switch(aux_stream):
            torch.npu.current_stream().wait_event(e1)
            weights_raw = impl.weights_proj(hidden_states[:actual_tokens])

        # Main stream: quant_scatter + scatter for both decode and prefill
        decode_q_quant = None
        decode_q_scale = None
        if has_decode:
            decode_q_quant, decode_q_scale, _, _ = impl._indexer_quant_scatter(q_idx, kv_idx, ik, isc, isc_meta, wp)

            torch.ops._C_ascend.npu_scatter_nd_update_v2(
                kv_cache[0],
                compressor_attn_metadata.decode.slot_mapping,
                decode_compressed_kv,
            )

        prefill_q_quant = None
        prefill_q_scale = None
        if has_prefill:
            prefill_q_quant, prefill_q_scale, _, _ = impl._indexer_quant_scatter(
                pq_idx, pkv_idx, pik, pisc, pisc_meta, pwp
            )

            torch.ops._C_ascend.npu_scatter_nd_update_v2(
                kv_cache[0],
                compressor_attn_metadata.prefill.slot_mapping,
                prefill_compressed_kv,
            )

        torch.npu.current_stream().wait_stream(aux_stream)

        scale = impl.indexer_softmax_scale * impl.indexer_heads**-0.5

        # Split weights into decode and prefill portions
        decode_weights = None
        prefill_weights = None
        if has_decode and has_prefill:
            decode_weights_raw = weights_raw[:decode_tokens]
            prefill_weights_raw = weights_raw[decode_tokens:actual_tokens]
            decode_weights = decode_weights_raw * scale
            prefill_weights = prefill_weights_raw * scale
        elif has_decode:
            decode_weights = weights_raw * scale
        elif has_prefill:
            prefill_weights = weights_raw * scale

        # ============================================================
        # Phase 3: QLI + sparse attention + o_proj (unified)
        # ============================================================
        impl.dsa_dual_stream_finish(
            self.dsa_attn.layer_name,
            output,
            decode_q=q if has_decode else None,
            decode_q_quant=decode_q_quant,
            decode_q_scale=decode_q_scale,
            decode_weights=decode_weights,
            prefill_q=pq if has_prefill else None,
            prefill_q_quant=prefill_q_quant,
            prefill_q_scale=prefill_q_scale,
            prefill_weights=prefill_weights,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            num_decode_tokens=decode_tokens,
            num_actual_tokens=actual_tokens,
        )
    else:
        self.dsa_attn.impl.forward(
            self.dsa_attn.layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output
        )
    return


def dsa_forward_fake(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="dsa_forward",
    op_func=dsa_forward,
    mutates_args=["output"],
    fake_impl=dsa_forward_fake,
    dispatch_key="PrivateUse1",
)


def filter_metadata(metadata, prefix):
    # filter using prefix, sort by key for deterministic order
    return [v for k, v in sorted(metadata.items()) if k.startswith(prefix)]


def _build_kv_cache(self, forward_context):
    """Construct the 6-tuple KV cache used by impl.forward()."""
    compress_kv_cache = None
    swa_kv_cache = self.swa_cache_layer.kv_cache
    state_cache = None
    indexer_state_cache = None
    indexer_k_cache = None
    indexer_scale_cache = None

    if self.compress_ratio > 1:
        state_cache = self.compressor.state_cache.kv_cache
        compress_kv_cache = self.dsa_attn.kv_cache
        virtual_engine = getattr(forward_context, "virtual_engine", None)
        if virtual_engine is not None and isinstance(compress_kv_cache, (list, tuple)):
            compress_kv_cache = compress_kv_cache[virtual_engine]
    if self.compress_ratio == 4:
        indexer_state_cache = self.indexer.compressor.state_cache.kv_cache
        indexer_k_cache, indexer_scale_cache = (
            self.indexer.k_cache.kv_cache[0][0],
            self.indexer.k_cache.kv_cache[0][1],
        )

    return tuple(
        [
            unfold_kvcache(cache)
            for cache in (
                compress_kv_cache,
                swa_kv_cache,
                state_cache,
                indexer_state_cache,
                indexer_k_cache,
                indexer_scale_cache,
            )
        ]
    )


def unfold_kvcache(kvcache):
    while isinstance(kvcache, list) and len(kvcache) == 1:
        kvcache = kvcache[0]
    return kvcache
