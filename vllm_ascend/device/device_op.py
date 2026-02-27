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

from vllm_ascend.quantization.mxfp_compat import (
    FLOAT4_E2M1FN_X2_DTYPE,
    FLOAT8_E8M0FNU_DTYPE,
    HIFLOAT8_DTYPE,
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
    def npu_dynamic_quant(
        hidden_states: torch.Tensor,
        dynamic_scale: torch.Tensor | None = None,
        *,
        act_quant_type=torch.float8_e4m3fn,
        use_mxfp_quant: bool = False,
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP8 MoE quantization is only supported on Ascend A5.")

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
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP8 MoE quantization is only supported on Ascend A5.")

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
            raise RuntimeError("MXFP8 MoE quantization is only supported on Ascend A5.")

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
            raise RuntimeError("MXFP8 MoE quantization is only supported on Ascend A5.")

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


class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key, value=value.contiguous(), key_cache=key_cache, value_cache=value_cache, slot_mapping=slot_mapping
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
            return torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=act_quant_type)

        if dynamic_scale.ndim == 2:
            dynamic_scale = dynamic_scale.reshape(dynamic_scale.shape[0], dynamic_scale.shape[1] // 2, 2)

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
            quant_dtype=torch.float8_e4m3fn,
            weight_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            x_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        )
        return out, out_scale, None

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

        quant_dtypes = tuple(dtype for dtype in (FLOAT4_E2M1FN_X2_DTYPE, HIFLOAT8_DTYPE) if dtype is not None)
        scale_dtypes = tuple(dtype for dtype in (FLOAT8_E8M0FNU_DTYPE,) if dtype is not None)

        output_dtype = (
            input_dtype
            if input_dtype in [torch.bfloat16, torch.float16]
            else (torch.bfloat16 if use_bf16 else torch.float16)
        )

        return {
            "scale_dtype": scale_type if scale_type in scale_dtypes else None,
            "per_token_scale_dtype": per_token_scale_type if per_token_scale_type in scale_dtypes else None,
            "x_dtype": act_quant_type if act_quant_type in quant_dtypes else None,
            "weight_dtype": weight_quant_type if weight_quant_type in quant_dtypes else None,
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


def get_device_adaptor():
    ascend_device_type = get_ascend_device_type()
    if ascend_device_type == AscendDeviceType.A5:
        return A5DeviceAdaptor
    return BaseDeviceAdaptor


DeviceOperator: type["BaseDeviceAdaptor"] | None = get_device_adaptor()
