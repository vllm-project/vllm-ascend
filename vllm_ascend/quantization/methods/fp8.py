#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config

from .base import QuantType
from .registry import register_scheme
from .w4a8_mxfp4 import AscendW4A8MXFPDynamicFusedMoEMethod
from .w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod


@register_scheme("FP8", "ds_linear")
class AscendW8A8MXFP8DSDynamicLinearMethod(AscendW8A8MXFP8DynamicLinearMethod):
    """Linear method for DS original W8A8 mxfp(blocksize: 128 * 128) quantization.

    scales are reorganize as blocksize 32 * 1 in process_weights_after_loading
    """

    model_dtype = None

    def __init__(self, quant_config):
        super().__init__()
        self.block_size = quant_config.get("weight_block_size", [128, 128])[0]
        vllm_config = get_current_vllm_config()
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        hf_config = vllm_config.model_config.hf_config
        self.n_groups = hf_config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.o_lora_rank = hf_config.o_lora_rank

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(
            output_size // self.block_size, input_size // self.block_size, dtype=torch.float32
        )
        params_dict["_packed_dim"] = 0
        params_dict["_packed_factor"] = self.block_size
        return params_dict

    def process_weights_after_loading(self, layer):
        layer.weight_scale.data = layer.weight_scale.data.view(torch.int32) >> 23 & 0xFF
        layer.weight_scale.data = layer.weight_scale.data.to(torch.uint8)
        layer.weight_scale.data = layer.weight_scale.data.repeat_interleave(4, dim=1).repeat_interleave(128, dim=0)
        n_dim, k_dim = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)

        if layer.prefix.endswith("wo_a"):
            layer.weight.data = (
                layer.weight.data.T.reshape(self.n_local_groups, self.o_lora_rank, -1).transpose(1, 2).contiguous()
            )
            layer.weight_scale.data = (
                layer.weight_scale.data.transpose(0, 1)
                .reshape(self.n_local_groups, self.o_lora_rank, -1, 2)
                .transpose(1, 2)
                .contiguous()
            )


@register_scheme("FP8", "w4a8_moe")
class AscendW4A8MXFPDSDynamicFusedMoEMethod(AscendW4A8MXFPDynamicFusedMoEMethod):
    """FusedMoe method for DS original w4a8 mxfp quantization."""

    model_dtype = None
    quant_type: QuantType = QuantType.W4A8MXFP

    def __init__(self, quant_config, tid2eid=None):
        super().__init__()
        self.tid2eid = tid2eid

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.group_size,
            dtype=torch.float8_e8m0fnu,
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.float8_e8m0fnu
        )
        return param_dict

    def process_weights_after_loading(self, layer):
        layer.w13_weight.data = torch_npu.npu_format_cast(
            layer.w13_weight.data.view(torch.uint8),
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )
        layer.w2_weight.data = torch_npu.npu_format_cast(
            layer.w2_weight.data.view(torch.uint8),
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        g, n, k = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = (
            layer.w13_weight_scale.data.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)
        )
        g, n, k = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = (
            layer.w2_weight_scale.data.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)
        )
