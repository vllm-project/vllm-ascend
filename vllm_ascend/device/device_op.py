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
import torch
import torch_npu

from vllm_ascend.device.mxfp_compat import (
    QUANT_DTYPES,
    SCALE_DTYPES,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


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
            return torch_npu.npu_dynamic_quant(hidden_states)

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
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        return torch_npu.npu_grouped_matmul_swiglu_quant(
            x=x,
            weight=weight,
            bias=bias,
            group_list=group_list,
            weight_scale=weight_scale,
            x_scale=x_scale,
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
    def mla_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_starts, key, value):
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


class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping.contiguous(),
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
    ):
        if not use_mxfp_quant:
            return BaseDeviceAdaptor.npu_grouped_matmul_swiglu_quant(
                x=x,
                weight=weight,
                group_list=group_list,
                weight_scale=weight_scale,
                x_scale=x_scale,
                bias=bias,
                use_mxfp_quant=False,
            )

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
            weight_scale_dtype=torch.float8_e8m0fnu,
            x_scale_dtype=torch.float8_e8m0fnu,
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
    ) -> torch.Tensor:
        if not use_mxfp_quant:
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
            scale_type=scale_type,
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
    def mla_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_offset, key, value):
        torch_npu.npu_gather_pa_kv_cache(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu,
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

        decode_q_nope, decode_q_pe, _, _, _ = torch_npu.npu_mla_prolog_v3(
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
            query_quant_mode=0,
            weight_quant_mode=3,
        )

        decode_q_nope = decode_q_nope.view(bsz, atten_obj.num_heads, atten_obj.kv_lora_rank)
        decode_q_pe = decode_q_pe.view(bsz, atten_obj.num_heads, -1)

        from vllm_ascend.attention.mla_v1 import DecodeMLAPreprocessResult

        decode_preprocess_res = DecodeMLAPreprocessResult(decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe)
        return decode_preprocess_res, None


def get_device_adaptor() -> type["BaseDeviceAdaptor"]:
    ascend_device_type = get_ascend_device_type()
    if ascend_device_type == AscendDeviceType.A5:
        return A5DeviceAdaptor
    return BaseDeviceAdaptor


DeviceOperator: type["BaseDeviceAdaptor"] = get_device_adaptor()
