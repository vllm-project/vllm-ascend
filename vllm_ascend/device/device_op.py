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
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    QUANT_DTYPES,
    SCALE_DTYPES,
)
from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel
from vllm_ascend.ops.triton.fla.solve_tril import solve_tril_16x16_kernel
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

if HAS_TRITON:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F811
else:
    triton_q_rms = None  # type: ignore


class BaseDeviceAdaptor:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu._npu_reshape_and_cache(
            key=key, value=value, key_cache=key_cache, value_cache=value_cache, slot_indices=slot_mapping
        )

    @staticmethod
    def npu_moe_init_routing(
        hidden_states,
        topk_ids,
        *,
        scale=None,
        active_num: int,
        expert_num: int,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        active_expert_range=None,
        quant_mode: int = -1,
    ):
        return torch.ops._C_ascend.npu_moe_init_routing_custom(
            hidden_states,
            topk_ids,
            scale=scale,
            active_num=active_num,
            expert_num=expert_num,
            expert_tokens_num_type=expert_tokens_num_type,
            expert_tokens_num_flag=expert_tokens_num_flag,
            active_expert_range=active_expert_range,
            quant_mode=quant_mode,
        )

    @staticmethod
    def maybe_normalize_mxfp_scale_layout(scale: torch.Tensor | None) -> torch.Tensor | None:
        return scale

    @staticmethod
    def moe_gating_top_k(
        x: torch.Tensor,
        *,
        k: int,
        k_group: int,
        group_count: int,
        group_select_mode: int,
        renorm: int,
        norm_type: int,
        out_flag: bool,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
        bias_opt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, out = torch.ops._C_ascend.moe_gating_top_k(
            x,
            k=k,
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps,
            bias_opt=bias_opt,
        )
        return topk_weights, topk_ids.to(torch.int32), out

    @staticmethod
    def npu_dynamic_quant(
        hidden_states: torch.Tensor,
        dynamic_scale: torch.Tensor | None = None,
        *,
        act_quant_type=torch.float8_e4m3fn,
        use_mxfp_quant: bool = False,
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        if dynamic_scale is None:
            return torch_npu.npu_dynamic_quant(hidden_states, dst_type=act_quant_type)

        return hidden_states, dynamic_scale

    @staticmethod
    def npu_grouped_matmul_swiglu_quant(
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scale: torch.Tensor,
        bias=None,
        use_mxfp_quant: bool = False,
        act_quant_type: torch.dtype | int = torch.float8_e4m3fn,
        weight_quant_type: torch.dtype | int = torch.float8_e4m3fn,
        swiglu_limit: float = 0.0,
        mxfp_quant_dtype: QuantType | None = None,
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        return torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz(
            x=x,
            weight=weight,
            weight_scale=weight_scale,
            x_scale=x_scale,
            group_list=group_list,
            bias=bias,
            swiglu_limit=swiglu_limit,
        )

    @staticmethod
    def get_quant_gmm2_kwargs(
        *,
        input_dtype: torch.dtype,
        act_quant_type,
        weight_quant_type,
        scale_type,
        per_token_scale_type,
        use_bf16: bool = True,
        use_mxfp_quant: bool = False,
    ) -> dict:
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        return {
            "output_dtype": input_dtype if input_dtype in [torch.bfloat16, torch.float16] else torch.bfloat16,
        }

    @classmethod
    def npu_grouped_matmul_gmm2(
        cls,
        *,
        hidden_states: torch.Tensor,
        weight: list[torch.Tensor] | torch.Tensor,
        weight_scale: list[torch.Tensor] | torch.Tensor,
        per_token_scale: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
        input_dtype: torch.dtype,
        act_quant_type,
        weight_quant_type,
        scale_type,
        per_token_scale_type,
        use_bf16: bool = True,
        use_mxfp_quant: bool = False,
        bias=None,
        fallback_output_dtype: torch.dtype | None = None,
        mxfp_quant_dtype: QuantType | None = None,
    ) -> torch.Tensor:
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        if fallback_output_dtype is None:
            fallback_output_dtype = weight_scale[0].dtype if isinstance(weight_scale, list) else weight_scale.dtype
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=weight,
            scale=weight_scale,
            bias=bias,
            per_token_scale=[per_token_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=fallback_output_dtype,
        )[0]

    @staticmethod
    def kv_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_starts, key, value):
        torch_npu.atb.npu_paged_cache_load(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu,
            seq_starts=seq_starts,
            key=key,
            value=value,
        )

    @staticmethod
    def mla_preprocess_only_decode(atten_obj, hidden_states, kv_cache, attn_metadata):
        bsz = attn_metadata.num_decode_tokens
        hidden_states = hidden_states[:bsz]

        cos_shape = attn_metadata.decode.cos.shape
        cos = attn_metadata.decode.cos.view(cos_shape[0], cos_shape[-1])
        sin = attn_metadata.decode.sin.view(cos_shape[0], cos_shape[-1])

        decode_k_nope, decode_k_pe = kv_cache[0], kv_cache[1]
        dequant_scale_q_nope = None
        if atten_obj.fa_quant_layer:
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe, dequant_scale_q_nope = torch_npu.npu_mla_prolog_v2(
                quantized_x,
                atten_obj.wd_q,
                atten_obj.wu_q,
                atten_obj.W_UK_T,
                atten_obj.wd_kv,
                atten_obj.gamma1,
                atten_obj.gamma2,
                sin,
                cos,
                attn_metadata.slot_mapping[:bsz].to(torch.int64),
                decode_k_nope,
                decode_k_pe,
                dequant_scale_x=pertoken_scale.view(-1, 1),
                dequant_scale_w_dq=atten_obj.dequant_scale_w_dq,
                dequant_scale_w_uq_qr=atten_obj.dequant_scale_w_uq_qr,
                dequant_scale_w_dkv_kr=atten_obj.dequant_scale_w_dkv_kr,
                quant_scale_ckv=atten_obj.quant_kscale,
                cache_mode="PA_NZ",
            )
        else:
            decode_q_nope = torch.empty(
                (hidden_states.shape[0], atten_obj.W_UK_T.shape[0], decode_k_nope.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            decode_q_pe = torch.empty(
                (hidden_states.shape[0], atten_obj.W_UK_T.shape[0], decode_k_pe.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            torch.ops._C_ascend.mla_preprocess(
                hidden_states,
                atten_obj.wd_qkv,
                atten_obj.deq_scale_qkv,
                atten_obj.gamma1,
                atten_obj.beta1,
                atten_obj.wu_q,
                atten_obj.qb_deq_scl,
                atten_obj.gamma2,
                cos,
                sin,
                atten_obj.W_UK_T,
                decode_k_nope,
                decode_k_pe,
                attn_metadata.slot_mapping[:bsz],
                quant_scale0=atten_obj.quant_scale0,
                quant_offset0=atten_obj.quant_offset0,
                bias0=atten_obj.quant_bias_qkv,
                quant_scale1=atten_obj.quant_scale1,
                quant_offset1=atten_obj.quant_offset1,
                bias1=atten_obj.qb_qt_bias,
                ctkv_scale=atten_obj.ctkv_scale,
                q_nope_scale=atten_obj.q_nope_scale,
                cache_mode="nzcache" if atten_obj.enable_kv_nz else "krope_ctkv",
                quant_mode="per_tensor_quant_asymm",
                q_out0=decode_q_nope,
                kv_cache_out0=decode_k_nope,
                q_out1=decode_q_pe,
                kv_cache_out1=decode_k_pe,
                enable_inner_out=False,
                inner_out=torch.tensor([], device=hidden_states.device),
            )
            decode_q_nope = decode_q_nope.view(bsz, atten_obj.num_heads, atten_obj.kv_lora_rank)
            decode_q_pe = decode_q_pe.view(bsz, atten_obj.num_heads, -1)

        decode_q_nope, decode_q_pe = atten_obj.reorg_decode_q(decode_q_nope, decode_q_pe)

        from vllm_ascend.attention.mla_v1 import DecodeMLAPreprocessResult

        decode_preprocess_res = DecodeMLAPreprocessResult(
            decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe, dequant_scale_q_nope=dequant_scale_q_nope
        )
        return decode_preprocess_res, None

    @staticmethod
    def sfa_preprocess_with_mlapo(
        sfa_impl,
        hidden_states: torch.Tensor,
        kv_cache: tuple,
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> tuple:
        k_nope, k_pe = kv_cache[0], kv_cache[1]
        ql_nope = torch.empty(
            (num_input_tokens, sfa_impl.W_UK_T.shape[0], k_nope.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_pe = torch.empty(
            (num_input_tokens, sfa_impl.W_UK_T.shape[0], k_pe.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_c = torch.empty(
            (num_input_tokens, sfa_impl.q_lora_rank),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._C_ascend.mla_preprocess(
            hidden_states,
            sfa_impl.wd_qkv,
            sfa_impl.deq_scale_qkv,
            sfa_impl.gamma1,
            sfa_impl.beta1,
            sfa_impl.wu_q,
            sfa_impl.qb_deq_scl,
            sfa_impl.gamma2,
            cos,
            sin,
            sfa_impl.W_UK_T,
            k_nope,
            k_pe,
            slot_mapping,
            quant_scale0=sfa_impl.quant_scale0,
            quant_offset0=sfa_impl.quant_offset0,
            bias0=sfa_impl.quant_bias_qkv,
            quant_scale1=sfa_impl.quant_scale1,
            quant_offset1=sfa_impl.quant_offset1,
            bias1=sfa_impl.qb_qt_bias,
            ctkv_scale=sfa_impl.ctkv_scale,
            q_nope_scale=sfa_impl.q_nope_scale,
            cache_mode="krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            enable_inner_out=True,
            q_out0=ql_nope,
            kv_cache_out0=k_nope,
            q_out1=q_pe,
            kv_cache_out1=k_pe,
            inner_out=q_c,
        )
        return hidden_states, ql_nope, q_pe, q_c

    @staticmethod
    def indexer_select_post_process(
        sfa_impl,
        q_li: torch.Tensor,
        q_li_scale: torch.Tensor | None,
        q_li_shape_ori: tuple[Any, ...] | None,
        weights: torch.Tensor,
        kv_cache: tuple,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        use_sparse_c8_indexer: bool,
        use_torch_npu_lightning_indexer: bool,
    ) -> torch.Tensor:
        # DSV3.2 currently has graph compilation issues when using torch_npu.npu.lightning_indexer.
        # So two branches are maintained temporarily.
        # TODO: torch.ops._C_ascend.npu_lightning_indexer needs to be removed.
        if sfa_impl.use_sparse_c8_indexer:
            assert len(kv_cache) == 4
            assert q_li_scale is not None
            assert q_li_shape_ori is not None
            weights = weights.to(torch.float16)
            topk_indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
                query=q_li.view(q_li_shape_ori),
                key=kv_cache[2],
                weights=weights,
                query_dequant_scale=q_li_scale.view(q_li_shape_ori[:-1]),
                key_dequant_scale=kv_cache[3].squeeze(2),  # B S N D -> B S D
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        elif sfa_impl.use_torch_npu_lightning_indexer:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q_li,
                key=kv_cache[2],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        else:
            topk_indices, _ = torch.ops._C_ascend.npu_lightning_indexer(
                query=q_li,
                key=kv_cache[2],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        return topk_indices

    @staticmethod
    def execute_sparse_flash_attention_process(
        sfa_impl,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple,
        topk_indices: torch.Tensor,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor:
        block_table = attn_metadata.block_table
        kv = kv_cache[0]
        key_rope = kv_cache[1]

        attn_output, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            scale_value=sfa_impl.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=key_rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
        )
        return attn_output

    @staticmethod
    def npu_flash_attention(query, key, value, seq_lens_cpu, head_num, scale_value, num_kv_heads):
        if query.dtype == torch.float32:
            # _npu_flash_attention_unpad does not support FP32.
            cumulative_seq_lens = seq_lens_cpu.cumsum(0).tolist()
            return torch_npu.npu_fusion_attention(
                query=query,
                key=key,
                value=value,
                actual_seq_qlen=cumulative_seq_lens,
                actual_seq_kvlen=cumulative_seq_lens,
                head_num=head_num,
                scale=scale_value,
                input_layout="TND",
            )[0]

        context_layer = torch.empty_like(query)

        torch_npu._npu_flash_attention_unpad(
            query=query,
            key=key,
            value=value,
            seq_len=seq_lens_cpu,
            scale_value=scale_value,
            num_heads=head_num,
            num_kv_heads=num_kv_heads,
            out=context_layer,
        )

        return context_layer

    # ===== Sparse Attention Metadata & Op Selectors =====

    @staticmethod
    def get_dsa_sparse_attn_metadata_op():
        """Returns the metadata-building operator for sparse attention."""
        return torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata

    @staticmethod
    def get_dsa_sparse_attn_metadata_kwargs(device):
        """Returns kwargs for sparse attention metadata builder."""
        return {"device": str(device)}

    @staticmethod
    def get_dsa_sparse_attn_op():
        """Returns the sparse attention operator."""
        return torch.ops._C_ascend.npu_sparse_attn_sharedkv

    @staticmethod
    def get_dsa_sparse_attn_base_kwargs():
        """Returns base kwargs for sparse attention (extended by caller)."""
        return {}

    # ===== SWA / Compressor KV Scatter =====

    @staticmethod
    def dsa_kv_compress_scatter(cache, x, slot_mapping):
        """Scatter KV into cache. Non-A5: simple scatter of pre-quantized tensor."""
        torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, x)

    # ===== Indexer Quant + Scatter =====

    @staticmethod
    def indexer_quantize_query(q):
        """Quantize indexer query for lightning_indexer.
        Non-A5: int8 quant with float16 scale."""
        q_quant, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=torch.int8)
        q_scale = q_scale.to(torch.float16)
        return q_quant, q_scale

    @staticmethod
    def indexer_quant_scatter(q, kv, indexer_k_cache, indexer_scale_cache, indexer_full_cache, slot_mapping):
        """Quantize q and scatter kv into indexer cache.
        Non-A5: int8 quant + 2x scatter_nd_update_v2 for k_cache and scale_cache."""
        q, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=torch.int8)
        q_scale = q_scale.to(torch.float16)

        kv_out = kv
        kv_scale_out = None
        if kv is not None:
            kv_out, kv_scale_out = torch_npu.npu_dynamic_quant(kv, dst_type=torch.int8)
            kv_scale_out = kv_scale_out.unsqueeze(-1).to(torch.float16)
            if kv_scale_out.ndim < 4:
                kv_scale_out = kv_scale_out.unsqueeze(-1)
            torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_k_cache, slot_mapping, kv_out)
            torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_scale_cache, slot_mapping, kv_scale_out)

        return q, q_scale, kv_out, kv_scale_out

    @staticmethod
    def indexer_quant_scatter_part1(kv, indexer_k_cache, indexer_full_cache, slot_mapping):
        """Part1 of multi-stream indexer scatter.
        Non-A5: quantize kv + scatter k_cache.
        Returns (kv_quant, kv_scale) for use in Part3, or (None, None) if kv is None."""
        if kv is None:
            return None, None
        kv_out, kv_scale = torch_npu.npu_dynamic_quant(kv, dst_type=torch.int8)
        kv_scale = kv_scale.unsqueeze(-1)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_k_cache, slot_mapping, kv_out)
        return kv_out, kv_scale

    @staticmethod
    def dsa_indexer_scatter_scale_part3(kv_scale, indexer_scale_cache, slot_mapping):
        """Part3 of multi-stream indexer scatter.
        Non-A5: scatter scale_cache (float16 conversion + scatter)."""
        kv_scale = kv_scale.to(torch.float16)
        if kv_scale.ndim < 4:
            kv_scale = kv_scale.unsqueeze(-1)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_scale_cache, slot_mapping, kv_scale)

    @staticmethod
    def warmup_indexer_quant_scatter(hidden_states, slot_mapping):
        """Warmup profiling for indexer quant+scatter.
        Non-A5: int8 quant + 2x scatter with dummy cache tensors."""
        kv_dummy, kv_scale_dummy = torch_npu.npu_dynamic_quant(hidden_states, dst_type=torch.int8)
        kv_scale_dummy = kv_scale_dummy.unsqueeze(-1).to(torch.float16)
        if kv_scale_dummy.ndim < 4:
            kv_scale_dummy = kv_scale_dummy.unsqueeze(-1)
        dummy_shape = (1, 1, 1, kv_dummy.shape[-1])
        indexer_k_cache = torch.zeros(dummy_shape, dtype=kv_dummy.dtype, device=hidden_states.device)
        indexer_scale_cache = torch.zeros(dummy_shape, dtype=torch.float16, device=hidden_states.device)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_k_cache, slot_mapping, kv_dummy)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_scale_cache, slot_mapping, kv_scale_dummy)

    # ===== Lightning Indexer Dtype Prep =====

    @staticmethod
    def prepare_dsa_indexer_weights(weights):
        """Non-A5: cast indexer weights to float16."""
        return weights.to(torch.float16)

    @staticmethod
    def prepare_dsa_indexer_query_scale(q_scale):
        """Non-A5: q_scale already float16, pass through."""
        return q_scale

    @staticmethod
    def prepare_dsa_indexer_key_scale(indexer_scale_cache):
        """Non-A5: cast key dequant scale to float16."""
        return indexer_scale_cache.squeeze(-2).to(torch.float16)

    # ===== Q RMS Norm =====

    @staticmethod
    def apply_dsa_q_rms(q, eps, q_norm_without_weight=None):
        """Apply Q RMS norm. Non-A5: triton_q_rms.
        A5: uses q_norm_without_weight callable when provided."""
        if triton_q_rms is not None:
            return triton_q_rms(q, eps)
        else:
            dtype = q.dtype
            q = q.float()
            variance = q.square().mean(-1, keepdim=True)
            q = q * torch.rsqrt(variance + eps)
            return q.to(dtype)

    # ===== KV Cache Helpers =====

    @staticmethod
    def unpack_dsa_indexer_kv_cache(kv_cache):
        """Unpack indexer kv_cache tuple.
        Non-A5: returns (state_cache, k_cache, scale_cache, None).
        A5: returns (state_cache, k_cache, scale_cache, full_cache)."""
        _, _, _, indexer_state_cache, indexer_k_cache, indexer_scale_cache = kv_cache
        return indexer_state_cache, indexer_k_cache, indexer_scale_cache, None

    @staticmethod
    def unpack_dsa_forward_kv_cache(kv_cache, compress_ratio):
        """Unpack kv_cache for forward pass.
        Returns 6-tuple: (compress_kv_cache, swa_kv_cache, state_cache,
        indexer_k_cache, indexer_scale_cache, indexer_full_cache).
        Non-A5: indexer_full_cache is always None.
        All devices: unused slots are None.
        """
        idx_full = 6  # 7th element (indexer_full_cache), A5 only
        full_cache = kv_cache[idx_full] if len(kv_cache) > idx_full else None
        if compress_ratio == 4:
            # [0]=compress, [1]=swa, [2]=state, [3]=unused, [4]=ik, [5]=isc
            return (kv_cache[0], kv_cache[1], kv_cache[2], kv_cache[4], kv_cache[5], full_cache)
        elif compress_ratio == 128:
            return (kv_cache[0], kv_cache[1], kv_cache[2], None, None, full_cache)
        else:
            return (None, kv_cache[1], None, None, None, full_cache)

    @staticmethod
    def pad_dsa_decode_slot_mapping(slot_mapping, num_decode_tokens, compress_ratio, num_decodes):
        """Pad slot_mapping for decode metadata. Non-A5: pass through."""
        return slot_mapping

    @staticmethod
    def format_dsa_slot_mapping(slot_mapping, block_size):
        """Format slot_mapping for metadata storage.
        Non-A5: 2D [block_idx, offset]; A5: 1D pass-through."""
        return torch.stack([slot_mapping // block_size, slot_mapping % block_size], axis=-1)

    @staticmethod
    def get_dsa_decode_cu_seqlens_cmp_kv(cmp_kv_tensor):
        """Non-A5: return the cached cu_seqlens_cmp_kv tensor.
        A5 override always returns None."""
        return cmp_kv_tensor

    @staticmethod
    def add_dsa_sparse_attn_extra_kwargs(extra_kwargs, **kwargs_to_add):
        """Non-A5: add extra kwargs for sparse attention. A5: no-op."""
        extra_kwargs.update(kwargs_to_add)

    @staticmethod
    def get_dsa_decode_cu_seqlens_ori_kv(
        decode_ratio_to_sas_metadata, cache_key, seq_lens, num_decodes, zero_i32, fallback_cu_seqlens
    ):
        """Non-A5: return fallback directly (self.cu_seqlens_ori_kv)."""
        return fallback_cu_seqlens

    @staticmethod
    def get_dsa_kernel_block_sizes():
        """Non-A5: return supported kernel block sizes."""
        return [8, 32, 128]

    @staticmethod
    def chunk_scaled_dot_kkt_fwd(
        num_core, bh_step, task_num, k, beta, g_cumsum, A, cu_seqlens, chunk_indices, T, B, H, Hg, K, BT, BK
    ):
        chunk_scaled_dot_kkt_fwd_kernel[(num_core,)](
            k=k,
            beta=beta,
            g_cumsum=g_cumsum,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            B=B,
            H=H,
            Hg=Hg,
            K=K,
            BT=BT,
            BK=BK,
            bh_step=bh_step,
            task_num=task_num,
            num_core=num_core,
            num_warps=8,
            num_stages=3,
            multibuffer=True,
        )

        return A

    @staticmethod
    def solve_tril_16x16(
        A,
        Ad,
        cu_seqlens,
        chunk_indices,
        T,
        H,
        BT,
        LARGE_BLOCK_T,
        NT,
        B,
    ):
        extract_slice_stride_1 = LARGE_BLOCK_T // 32
        solve_tril_16x16_kernel[NT, B * H](
            A=A,
            Ad=Ad,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            LARGE_BLOCK_T=LARGE_BLOCK_T,
            EXTRACT_SLICE_STRIDE_1=extract_slice_stride_1,
            num_warps=1,
            num_stages=4,
        )

        return Ad

    @staticmethod
    def npu_gemma_rms_norm(x, weight, variance_epsilon):
        x, _ = torch.ops._C_ascend.npu_gemma_rms_norm(x, weight, variance_epsilon)
        return x

    @staticmethod
    def fused_gdn_gating(A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor, dt_bias: torch.Tensor):
        return torch.ops._C_ascend.npu_fused_gdn_gating(A_log, a, b, dt_bias.to(torch.float32))

    @staticmethod
    def split_qkv_rmsnorm_rope(
        input,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_bias,
        k_bias,
        cos_sin_cache,
        positions,
    ):
        results = torch.ops.vllm.qkv_rmsnorm_rope(
            input=input,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=q_hidden_size,
            kv_hidden_size=kv_hidden_size,
            head_dim=head_dim,
            eps=eps,
            q_bias=q_bias,
            k_bias=k_bias,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
        )
        return results


class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping.contiguous(),
            cache_mode="Norm",
        )

    @staticmethod
    def npu_moe_init_routing(
        hidden_states,
        topk_ids,
        *,
        scale=None,
        active_num: int,
        expert_num: int,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        active_expert_range=None,
        quant_mode: int = -1,
    ):
        return torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            scale=scale,
            active_num=active_num,
            expert_num=expert_num,
            expert_tokens_num_type=expert_tokens_num_type,
            expert_tokens_num_flag=expert_tokens_num_flag,
            active_expert_range=active_expert_range,
            quant_mode=quant_mode,
        )

    @staticmethod
    def maybe_normalize_mxfp_scale_layout(scale: torch.Tensor | None) -> torch.Tensor | None:
        if scale is None or scale.ndim != 2:
            return scale
        if scale.shape[-1] % 2 != 0:
            raise ValueError(f"Invalid MXFP scale shape: {tuple(scale.shape)}")
        return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)

    @staticmethod
    def moe_gating_top_k(
        x: torch.Tensor,
        *,
        k: int,
        k_group: int,
        group_count: int,
        group_select_mode: int,
        renorm: int,
        norm_type: int,
        out_flag: bool,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
        bias_opt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, out = torch_npu.npu_moe_gating_top_k(
            x,
            k=k,
            bias=bias_opt,
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=0,
            norm_type=norm_type,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps,
        )
        if norm_type == 0 and renorm == 1:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids.to(torch.int32), out

    @staticmethod
    def npu_dynamic_quant(
        hidden_states: torch.Tensor,
        dynamic_scale: torch.Tensor | None = None,
        *,
        act_quant_type=torch.float8_e4m3fn,
        use_mxfp_quant: bool = False,
    ):
        if not use_mxfp_quant:
            return BaseDeviceAdaptor.npu_dynamic_quant(
                hidden_states,
                dynamic_scale,
                act_quant_type=act_quant_type,
                use_mxfp_quant=False,
            )

        if dynamic_scale is None:
            hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=act_quant_type)

        return hidden_states, A5DeviceAdaptor.maybe_normalize_mxfp_scale_layout(dynamic_scale)

    @staticmethod
    def npu_grouped_matmul_swiglu_quant(
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scale: torch.Tensor,
        bias=None,
        use_mxfp_quant: bool = False,
        act_quant_type: torch.dtype | int = torch.float8_e4m3fn,
        weight_quant_type: torch.dtype | int = torch.float8_e4m3fn,
        swiglu_limit: float = 0.0,
        mxfp_quant_dtype: QuantType | None = None,
    ):
        if not use_mxfp_quant:
            if act_quant_type == torch.float8_e4m3fn:
                out, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                    x=x,
                    weight=[weight],
                    weight_scale=[weight_scale],
                    x_scale=x_scale,
                    group_list=group_list,
                    quant_dtype=torch.float8_e4m3fn,
                    dequant_dtype=torch.float32,
                )
                return out, out_scale, None
            else:
                return torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                    x=x,
                    weight=weight,
                    group_list=group_list,
                    weight_scale=weight_scale,
                    x_scale=x_scale,
                    bias=bias,
                    swiglu_limit=swiglu_limit,
                    use_mxfp_quant=False,
                )

        # W4A8 mxfp
        if mxfp_quant_dtype == QuantType.W4A8MXFP:
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[x],
                weight=[weight],
                scale=None,
                antiquant_scale=[weight_scale],
                scale_dtype=None,
                per_token_scale=[x_scale],
                per_token_scale_dtype=torch.float8_e8m0fnu,
                split_item=2,
                group_type=0,
                group_list=group_list,
                x_dtype=torch.float8_e4m3fn,
                weight_dtype=torch_npu.float4_e2m1fn_x2,
                output_dtype=torch.bfloat16,
            )[0]
            # DSV4 need swiglu_limit input
            out, out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                hidden_states,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=2,
                clamp_value=swiglu_limit,
            )
        else:
            out, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                x=x,
                weight=[weight],
                group_list=group_list,
                weight_scale=[weight_scale],
                x_scale=x_scale,
                dequant_mode=2,
                quant_mode=2,
                dequant_dtype=torch.float32,
                quant_dtype=act_quant_type,
                x_dtype=act_quant_type if act_quant_type in QUANT_DTYPES else None,
                weight_dtype=weight_quant_type if weight_quant_type in QUANT_DTYPES else None,
                weight_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                x_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            )
        return out, A5DeviceAdaptor.maybe_normalize_mxfp_scale_layout(out_scale), None

    @staticmethod
    def get_quant_gmm2_kwargs(
        *,
        input_dtype: torch.dtype,
        act_quant_type,
        weight_quant_type,
        scale_type,
        per_token_scale_type,
        use_bf16: bool = True,
        use_mxfp_quant: bool = False,
    ) -> dict:
        if not use_mxfp_quant:
            return BaseDeviceAdaptor.get_quant_gmm2_kwargs(
                input_dtype=input_dtype,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
                scale_type=scale_type,
                per_token_scale_type=per_token_scale_type,
                use_bf16=use_bf16,
                use_mxfp_quant=False,
            )

        output_dtype = (
            input_dtype
            if input_dtype in [torch.bfloat16, torch.float16]
            else (torch.bfloat16 if use_bf16 else torch.float16)
        )

        return {
            "scale_dtype": scale_type if scale_type in SCALE_DTYPES else None,
            "per_token_scale_dtype": per_token_scale_type if per_token_scale_type in SCALE_DTYPES else None,
            "x_dtype": act_quant_type if act_quant_type in QUANT_DTYPES else None,
            "weight_dtype": weight_quant_type if weight_quant_type in QUANT_DTYPES else None,
            "output_dtype": output_dtype,
        }

    @classmethod
    def npu_grouped_matmul_gmm2(
        cls,
        *,
        hidden_states: torch.Tensor,
        weight: list[torch.Tensor] | torch.Tensor,
        weight_scale: list[torch.Tensor] | torch.Tensor,
        per_token_scale: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
        input_dtype: torch.dtype,
        act_quant_type,
        weight_quant_type,
        scale_type,
        per_token_scale_type,
        use_bf16: bool = True,
        use_mxfp_quant: bool = False,
        bias=None,
        fallback_output_dtype: torch.dtype | None = None,
        mxfp_quant_dtype: QuantType | None = None,
    ) -> torch.Tensor:
        if not use_mxfp_quant:
            if act_quant_type == torch.float8_e4m3fn:
                fallback_output_dtype = torch.bfloat16
            return BaseDeviceAdaptor.npu_grouped_matmul_gmm2(
                hidden_states=hidden_states,
                weight=weight,
                weight_scale=weight_scale,
                per_token_scale=per_token_scale,
                group_list=group_list,
                group_list_type=group_list_type,
                input_dtype=input_dtype,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
                scale_type=scale_type,
                per_token_scale_type=per_token_scale_type,
                use_bf16=use_bf16,
                use_mxfp_quant=False,
                bias=bias,
                fallback_output_dtype=fallback_output_dtype,
            )

        gmm2_kwargs = cls.get_quant_gmm2_kwargs(
            input_dtype=input_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type if mxfp_quant_dtype != QuantType.W4A8MXFP else None,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=True,
        )
        output_dtype = gmm2_kwargs.pop("output_dtype")

        if isinstance(weight, list) and len(weight) != 1:
            raise ValueError(f"w2 must have a single tensor in MXFP path, but got {len(weight)}.")
        if isinstance(weight_scale, list) and len(weight_scale) != 1:
            raise ValueError(f"w2_scale must have a single tensor in MXFP path, but got {len(weight_scale)}.")
        gmm2_weight = weight if isinstance(weight, list) else [weight]
        gmm2_scale = weight_scale if isinstance(weight_scale, list) else [weight_scale]

        if mxfp_quant_dtype == QuantType.W4A8MXFP:
            gmm2_scale = None  # type: ignore[assignment]
            gmm2_kwargs.update({"antiquant_scale": [weight_scale]})

        if mxfp_quant_dtype == QuantType.W4A8MXFP:
            gmm2_scale = None  # type: ignore[assignment]
            gmm2_kwargs.update({"antiquant_scale": [weight_scale]})

        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=gmm2_weight,
            scale=gmm2_scale,
            bias=bias,
            per_token_scale=[per_token_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
            **gmm2_kwargs,
        )[0]

    @staticmethod
    def kv_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_offset, key, value):
        torch_npu.npu_gather_pa_kv_cache(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu.contiguous(),
            seq_offset=seq_offset,
            key=key,
            value=value,
        )

    @staticmethod
    def mla_preprocess_only_decode(atten_obj, hidden_states, kv_cache, attn_metadata):
        bsz = attn_metadata.num_decode_tokens
        hidden_states = hidden_states[:bsz].unsqueeze(1)
        hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
        dynamic_scale = dynamic_scale.reshape(hidden_states.shape[0] * hidden_states.shape[1], -1)
        cos_shape = attn_metadata.decode.cos.shape
        cos = attn_metadata.decode.cos.view(cos_shape[0], 1, cos_shape[-1])
        sin = attn_metadata.decode.sin.view(cos_shape[0], 1, cos_shape[-1])
        decode_k_nope, decode_k_pe = kv_cache[0], kv_cache[1]
        decode_q_nope, decode_q_pe, dequant_scale_q_nope, _, _ = torch_npu.npu_mla_prolog_v3(
            token_x=hidden_states,
            weight_dq=atten_obj.weight_dq,
            weight_uq_qr=atten_obj.weight_uq_qr,
            weight_uk=atten_obj.W_UK_T,
            weight_dkv_kr=atten_obj.weight_dkv_kr,
            rmsnorm_gamma_cq=atten_obj.q_a_layernorm.weight.data,
            rmsnorm_gamma_ckv=atten_obj.kv_a_layernorm.weight.data,
            rope_sin=sin,
            rope_cos=cos,
            kv_cache=decode_k_nope,
            kr_cache=decode_k_pe,
            cache_index=attn_metadata.slot_mapping[:bsz].view(bsz, -1).to(torch.int64),
            dequant_scale_x=dynamic_scale.view(torch.float8_e8m0fnu),
            dequant_scale_w_dq=atten_obj.weight_dq_scale.view(torch.float8_e8m0fnu),
            dequant_scale_w_uq_qr=atten_obj.weight_uq_qr_scale.view(torch.float8_e8m0fnu),
            dequant_scale_w_dkv_kr=atten_obj.weight_dkv_kr_scale.view(torch.float8_e8m0fnu),
            cache_mode="PA_BSND",
            query_quant_mode=1 if atten_obj.fa_quant_layer else 0,
            weight_quant_mode=3,
            kv_cache_quant_mode=1 if atten_obj.fa_quant_layer else 0,
            quant_scale_ckv=atten_obj.fak_descale_reciprocal if atten_obj.fa_quant_layer else None,
        )
        decode_q_nope = decode_q_nope.view(bsz, atten_obj.num_heads, atten_obj.kv_lora_rank)
        decode_q_pe = decode_q_pe.view(bsz, atten_obj.num_heads, -1)

        decode_q_nope, decode_q_pe = atten_obj.reorg_decode_q(decode_q_nope, decode_q_pe)
        from vllm_ascend.attention.mla_v1 import DecodeMLAPreprocessResult

        decode_preprocess_res = DecodeMLAPreprocessResult(
            decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe, dequant_scale_q_nope=dequant_scale_q_nope
        )
        return decode_preprocess_res, None

    # ===== Sparse Attention Metadata & Op Selectors =====

    @staticmethod
    def get_dsa_sparse_attn_metadata_op():
        return torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata

    @staticmethod
    def get_dsa_sparse_attn_metadata_kwargs(device):
        return {"kv_quant_mode": 1}

    @staticmethod
    def get_dsa_sparse_attn_op():
        return torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv

    @staticmethod
    def get_dsa_sparse_attn_base_kwargs():
        return {"kv_quant_mode": 1, "tile_size": 64, "rope_head_dim": 64}

    # ===== SWA / Compressor KV Scatter =====

    @staticmethod
    def dsa_kv_compress_scatter(cache, x, slot_mapping):
        """Scatter KV into cache with fused quantization+compression.
        A5: kv_compress_epilog handles quant/compress/scatter internally.
        Input x is unquantized bf16; cache shape is [..., head_dim]."""
        torch.ops._C_ascend.kv_compress_epilog(
            kv_compress_cache=cache.view(-1, 1, cache.shape[-1]),
            x=x.view(-1, x.shape[-1]),
            slot_mapping=slot_mapping,
            quant_group_size=64,
            quant_mode=2,
            round_scale_flag=True,
            layout=1,
        )

    # ===== Indexer Quant + Scatter =====

    @staticmethod
    def indexer_quantize_query(q):
        """Quantize indexer query. A5: fp8 quant, no extra scale conversion."""
        q_quant, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=torch.float8_e4m3fn)
        return q_quant, q_scale

    @staticmethod
    def indexer_quant_scatter(q, kv, indexer_k_cache, indexer_scale_cache, indexer_full_cache, slot_mapping):
        """Quantize q (fp8) and scatter kv via fused indexer_compress_epilog_v2.
        On A5, the fused op handles kv quantization, k_cache scatter, and
        scale_cache scatter internally. q is quantized separately for use
        by lightning_indexer."""
        q, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=torch.float8_e4m3fn)

        kv_out = kv
        kv_scale_out = None
        if kv is not None:
            torch.ops._C_ascend.indexer_compress_epilog_v2(
                indexer_compress_cache=indexer_full_cache.view(torch.uint8),
                x=kv,
                slot_mapping=slot_mapping,
                layout=2,
            )

        return q, q_scale, kv_out, kv_scale_out

    @staticmethod
    def indexer_quant_scatter_part1(kv, indexer_k_cache, indexer_full_cache, slot_mapping):
        """Part1 of multi-stream indexer scatter.
        A5: fused indexer_compress_epilog_v2 handles both k_cache and scale_cache.
        Returns (kv, None) to signal Part3 is a no-op."""
        if kv is None:
            return None, None
        torch.ops._C_ascend.indexer_compress_epilog_v2(
            indexer_compress_cache=indexer_full_cache.view(torch.uint8),
            x=kv,
            slot_mapping=slot_mapping,
            layout=2,
        )
        return kv, None

    @staticmethod
    def dsa_indexer_scatter_scale_part3(kv_scale, indexer_scale_cache, slot_mapping):
        """Part3 of multi-stream indexer scatter.
        A5: no-op — fused op in Part1 already handled scale_cache."""
        pass

    @staticmethod
    def warmup_indexer_quant_scatter(hidden_states, slot_mapping):
        """Warmup profiling for indexer quant+scatter.
        A5: fused indexer_compress_epilog_v2 with dummy cache tensor."""
        dummy_cache_shape = (1, 1, 1, hidden_states.shape[-1])
        indexer_full_cache_dummy = torch.zeros(dummy_cache_shape, dtype=torch.uint8, device=hidden_states.device)
        torch.ops._C_ascend.indexer_compress_epilog_v2(
            indexer_compress_cache=indexer_full_cache_dummy,
            x=hidden_states,
            slot_mapping=slot_mapping,
            layout=2,
        )

    # ===== Lightning Indexer Dtype Prep =====

    @staticmethod
    def prepare_dsa_indexer_weights(weights):
        """A5: cast indexer weights to float32 (fp8 scale format needs float)."""
        return weights.float()

    @staticmethod
    def prepare_dsa_indexer_query_scale(q_scale):
        """A5: cast query dequant scale to float32."""
        return q_scale.float()

    @staticmethod
    def prepare_dsa_indexer_key_scale(indexer_scale_cache):
        """A5: cast key dequant scale to float32."""
        return indexer_scale_cache.squeeze(-2).float()

    # ===== Q RMS Norm =====

    @staticmethod
    def apply_dsa_q_rms(q, eps, q_norm_without_weight=None):
        """Apply Q RMS norm. A5: uses q_norm_without_weight callable."""
        if q_norm_without_weight is not None:
            return q_norm_without_weight(q)

        if triton_q_rms is not None:
            return triton_q_rms(q, eps)
        else:
            dtype = q.dtype
            q = q.float()
            variance = q.square().mean(-1, keepdim=True)
            q = q * torch.rsqrt(variance + eps)
            return q.to(dtype)

    # ===== KV Cache Helpers =====

    @staticmethod
    def unpack_dsa_indexer_kv_cache(kv_cache):
        """Unpack indexer kv_cache tuple.
        A5: returns (state_cache, k_cache, scale_cache, full_cache)."""
        _, _, _, indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache = kv_cache
        return indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache

    @staticmethod
    def unpack_dsa_forward_kv_cache(kv_cache, compress_ratio):
        """Unpack kv_cache for forward pass. A5: 7-element tuple with
        indexer_full_cache at position 6; non-A5: 6 elements (None substituted)."""
        idx_full = 6
        full_cache = kv_cache[idx_full]
        if compress_ratio == 4:
            return (kv_cache[0], kv_cache[1], kv_cache[2], kv_cache[4], kv_cache[5], full_cache)
        elif compress_ratio == 128:
            return (kv_cache[0], kv_cache[1], kv_cache[2], None, None, full_cache)
        else:
            return (None, kv_cache[1], None, None, None, full_cache)

    @staticmethod
    def pad_dsa_decode_slot_mapping(slot_mapping, num_decode_tokens, compress_ratio, num_decodes):
        """A5: pad slot_mapping to target shape for ACL graph compatibility."""
        tmp = compress_ratio if compress_ratio != 0 else 1
        target_shape = min(num_decode_tokens, num_decode_tokens // tmp + num_decodes)
        pad_size = target_shape - slot_mapping.shape[0]
        if pad_size > 0:
            if slot_mapping.ndim == 1:
                slot_mapping = F.pad(slot_mapping, (0, pad_size), value=-1)
            else:
                slot_mapping = F.pad(slot_mapping, (0, 0, 0, pad_size), value=-1)
        else:
            slot_mapping = slot_mapping[:target_shape]
        return slot_mapping

    @staticmethod
    def format_dsa_slot_mapping(slot_mapping, block_size):
        """A5: 1D pass-through."""
        return slot_mapping

    @staticmethod
    def get_dsa_decode_cu_seqlens_cmp_kv(cmp_kv_tensor):
        """A5: cu_seqlens_cmp_kv is always None."""
        return None

    @staticmethod
    def add_dsa_sparse_attn_extra_kwargs(extra_kwargs, **kwargs_to_add):
        """A5: no-op — A5 ops do not need extra kwargs from this path."""
        pass

    @staticmethod
    def get_dsa_decode_cu_seqlens_ori_kv(
        decode_ratio_to_sas_metadata, cache_key, seq_lens, num_decodes, zero_i32, fallback_cu_seqlens
    ):
        """A5: compute from cumsum of seq_lens, with caching."""
        if decode_ratio_to_sas_metadata is not None and cache_key in decode_ratio_to_sas_metadata:
            return decode_ratio_to_sas_metadata[cache_key]
        cu_seqlens = torch.cat(
            [
                zero_i32,
                torch.cumsum(seq_lens[:num_decodes], dim=0).to(torch.int32),
            ]
        )
        if decode_ratio_to_sas_metadata is not None:
            decode_ratio_to_sas_metadata[cache_key] = cu_seqlens
        return cu_seqlens

    @staticmethod
    def get_dsa_kernel_block_sizes():
        """A5: return supported kernel block sizes."""
        return [8, 16, 128]

    @staticmethod
    def sfa_preprocess_with_mlapo(
        sfa_impl,
        hidden_states: torch.Tensor,
        kv_cache: tuple,
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> tuple:
        bsz = num_input_tokens
        slot_mapping = slot_mapping[:bsz]
        hidden_states_temp = hidden_states[:bsz].unsqueeze(1)
        cos = cos[:bsz, ...]
        sin = sin[:bsz, ...]

        is_quantized = getattr(sfa_impl, "mlapo_is_quantized", True)

        cos_shape = cos.shape
        cos = cos.view(cos_shape[0], 1, cos_shape[-1])
        sin = sin.view(cos_shape[0], 1, cos_shape[-1])

        decode_k_nope = kv_cache[0]
        use_c8 = getattr(sfa_impl, "use_sparse_c8_indexer", False)
        kr_cache = (
            torch.zeros(0, 0, decode_k_nope.shape[-2], cos_shape[-1], dtype=torch.bfloat16, device=decode_k_nope.device)
            if use_c8
            else kv_cache[1]
        )

        if is_quantized:
            hidden_states_temp, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
                hidden_states_temp, dst_type=torch.float8_e4m3fn
            )
            dynamic_scale = dynamic_scale.reshape(hidden_states_temp.shape[0] * hidden_states_temp.shape[1], -1)

            decode_q_nope, q_pe, _, q_c, q_c_scale = torch_npu.npu_mla_prolog_v3(
                token_x=hidden_states_temp,
                weight_dq=sfa_impl.weight_dq,
                weight_uq_qr=sfa_impl.weight_uq_qr,
                weight_uk=sfa_impl.W_UK_T,
                weight_dkv_kr=sfa_impl.weight_dkv_kr,
                rmsnorm_gamma_cq=sfa_impl.q_a_layernorm.weight.data,
                rmsnorm_gamma_ckv=sfa_impl.kv_a_layernorm.weight.data,
                rope_sin=sin,
                rope_cos=cos,
                kv_cache=decode_k_nope,
                kr_cache=kr_cache,
                cache_index=slot_mapping[:bsz].view(bsz, -1).to(torch.int64),
                dequant_scale_x=dynamic_scale.view(torch.float8_e8m0fnu),
                dequant_scale_w_dq=sfa_impl.weight_dq_scale.view(torch.float8_e8m0fnu),
                dequant_scale_w_uq_qr=sfa_impl.weight_uq_qr_scale.view(torch.float8_e8m0fnu),
                dequant_scale_w_dkv_kr=sfa_impl.weight_dkv_kr_scale.view(torch.float8_e8m0fnu),
                cache_mode="PA_BSND",
                weight_quant_mode=3,
                kv_cache_quant_mode=3 if use_c8 else 0,
                query_quant_mode=0,
                ckvkr_repo_mode=1 if use_c8 else 0,
                quant_scale_repo_mode=1 if use_c8 else 0,
                query_norm_flag=True,
            )

            decode_q_nope = decode_q_nope.view(bsz, sfa_impl.num_heads, sfa_impl.kv_lora_rank)
            q_pe = q_pe.view(bsz, sfa_impl.num_heads, -1)
            q_c = q_c.view(-1, q_c.shape[-1])
            q_c_scale = q_c_scale.view(-1, q_c_scale.shape[-1])
            return hidden_states, decode_q_nope, q_pe, (q_c, q_c_scale)
        else:
            decode_q_nope, q_pe, _, q_c, _ = torch_npu.npu_mla_prolog_v3(
                token_x=hidden_states_temp,
                weight_dq=sfa_impl.weight_dq,
                weight_uq_qr=sfa_impl.weight_uq_qr,
                weight_uk=sfa_impl.W_UK_T,
                weight_dkv_kr=sfa_impl.weight_dkv_kr,
                rmsnorm_gamma_cq=sfa_impl.q_a_layernorm.weight.data,
                rmsnorm_gamma_ckv=sfa_impl.kv_a_layernorm.weight.data,
                rope_sin=sin,
                rope_cos=cos,
                kv_cache=decode_k_nope,
                kr_cache=kr_cache,
                cache_index=slot_mapping[:bsz].view(bsz, -1).to(torch.int64),
                cache_mode="PA_BSND",
                weight_quant_mode=0,
                kv_cache_quant_mode=0,
                query_quant_mode=0,
                ckvkr_repo_mode=0,
                quant_scale_repo_mode=0,
                query_norm_flag=True,
            )

            decode_q_nope = decode_q_nope.view(bsz, sfa_impl.num_heads, sfa_impl.kv_lora_rank)
            q_pe = q_pe.view(bsz, sfa_impl.num_heads, -1)
            q_c = q_c.view(-1, q_c.shape[-1])
            return hidden_states, decode_q_nope, q_pe, q_c

    @staticmethod
    def indexer_select_post_process(
        sfa_impl,
        q_li: torch.Tensor,
        q_li_scale: torch.Tensor | None,
        q_li_shape_ori: tuple[Any, ...] | None,
        weights: torch.Tensor,
        kv_cache: tuple,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        use_sparse_c8_indexer: bool,
        use_torch_npu_lightning_indexer: bool,
    ) -> torch.Tensor:
        if use_sparse_c8_indexer:
            assert len(kv_cache) == 3
            assert q_li_shape_ori is not None

            if q_li_scale is not None:
                q_li_scale = q_li_scale.view(q_li_shape_ori[:-1])
                key_dequant_scale = kv_cache[2].squeeze(2)

                topk_indices = torch_npu.npu_quant_lightning_indexer(
                    query=q_li.view(q_li_shape_ori),
                    key=kv_cache[1],
                    weights=weights,
                    query_dequant_scale=q_li_scale,
                    key_dequant_scale=key_dequant_scale,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    actual_seq_lengths_key=actual_seq_lengths_key,
                    block_table=attn_metadata.block_table,
                    query_quant_mode=0,
                    key_quant_mode=0,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=2048,
                    sparse_mode=3,
                )
            else:
                topk_indices, _ = torch_npu.npu_lightning_indexer(
                    query=q_li.view(q_li_shape_ori),
                    key=kv_cache[1],
                    weights=weights,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    actual_seq_lengths_key=actual_seq_lengths_key,
                    block_table=attn_metadata.block_table,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=2048,
                    sparse_mode=3,
                )
        else:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q_li,
                key=kv_cache[2],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        return topk_indices

    @staticmethod
    def execute_sparse_flash_attention_process(
        sfa_impl,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple,
        topk_indices: torch.Tensor,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor:
        block_table = attn_metadata.block_table
        kv = kv_cache[0]
        key_rope = kv_cache[1]

        if kv.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            query = torch.cat([ql_nope, q_pe], dim=-1)

            attn_output = torch_npu.npu_kv_quant_sparse_flash_attention(
                query=query,
                key=kv,
                value=kv,
                sparse_indices=topk_indices,
                scale_value=sfa_impl.scale,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_key,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                key_quant_mode=2,
                value_quant_mode=2,
                rope_head_dim=64,
            )
        else:
            attn_output, _, _ = torch_npu.npu_sparse_flash_attention(
                query=ql_nope,
                key=kv,
                value=kv,
                sparse_indices=topk_indices,
                scale_value=sfa_impl.scale,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_key,
                query_rope=q_pe,
                key_rope=key_rope,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
            )
        return attn_output

    @staticmethod
    def npu_flash_attention(query, key, value, seq_lens_cpu, head_num, scale_value, num_kv_heads):
        cumulative_seq_lens = seq_lens_cpu.cumsum(0).tolist()

        context_layer = torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            actual_seq_qlen=cumulative_seq_lens,
            actual_seq_kvlen=cumulative_seq_lens,
            head_num=head_num,
            scale=scale_value,
            input_layout="TND",
        )[0]

        return context_layer

    @staticmethod
    def chunk_scaled_dot_kkt_fwd(
        num_core, bh_step, task_num, k, beta, g_cumsum, A, cu_seqlens, chunk_indices, T, B, H, Hg, K, BT, BK
    ):
        chunk_scaled_dot_kkt_fwd_kernel[(num_core,)](
            k=k,
            beta=beta,
            g_cumsum=g_cumsum,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            B=B,
            H=H,
            Hg=Hg,
            K=K,
            BT=BT,
            BK=BK,
            bh_step=bh_step,
            task_num=task_num,
            num_core=num_core,
            num_warps=8,
            num_stages=3,
            multibuffer=True,
            disable_tightly_coupled_buffer_reuse=True,
        )
        return A

    @staticmethod
    def solve_tril_16x16(
        A,
        Ad,
        cu_seqlens,
        chunk_indices,
        T,
        H,
        BT,
        LARGE_BLOCK_T,
        NT,
        B,
    ):
        solve_tril_16x16_kernel[NT, B * H](
            A=A,
            Ad=Ad,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            LARGE_BLOCK_T=LARGE_BLOCK_T,
            EXTRACT_SLICE_STRIDE_1=1,
            num_warps=1,
            num_stages=4,
        )

        return Ad

    @staticmethod
    def npu_gemma_rms_norm(x, weight, variance_epsilon):
        x, _ = torch_npu.npu_rms_norm(x, 1.0 + weight, variance_epsilon)
        return x

    @staticmethod
    def fused_gdn_gating(A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor, dt_bias: torch.Tensor):
        return fused_gdn_gating_patch(A_log, a, b, dt_bias)

    @staticmethod
    def split_qkv_rmsnorm_rope(
        input,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_bias,
        k_bias,
        cos_sin_cache,
        positions,
    ):
        results = torch.ops.vllm.qkv_rmsnorm_rope_simt(
            input=input,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=q_hidden_size,
            kv_hidden_size=kv_hidden_size,
            head_dim=head_dim,
            eps=eps,
            q_bias=q_bias,
            k_bias=k_bias,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
        )
        return results


def get_device_adaptor() -> type["BaseDeviceAdaptor"]:
    ascend_device_type = get_ascend_device_type()
    if ascend_device_type == AscendDeviceType.A5:
        return A5DeviceAdaptor
    return BaseDeviceAdaptor


DeviceOperator: type["BaseDeviceAdaptor"] = get_device_adaptor()
