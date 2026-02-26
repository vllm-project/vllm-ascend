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
    def quant_apply_mlp(**kwargs):
        from vllm_ascend.ops.fused_moe.moe_mlp import quant_apply_mlp as _impl

        return _impl(**kwargs)

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


class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key, value=value.contiguous(), key_cache=key_cache, value_cache=value_cache, slot_mapping=slot_mapping
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


def get_device_adaptor():
    ascend_device_type = get_ascend_device_type()
    if ascend_device_type == AscendDeviceType.A5:
        return A5DeviceAdaptor
    return BaseDeviceAdaptor


DeviceOperator: type["BaseDeviceAdaptor"] | None = get_device_adaptor()
