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

from typing import Any, Dict, cast

import torch
from vllm.config import get_current_vllm_config

from .registry import register_scheme
from .w8a8_dynamic import (AscendW8A8DynamicFusedMoEMethod,
                           AscendW8A8DynamicLinearMethod)
from .w8a8_static import AscendW8A8LinearMethod


@register_scheme("W8A8_MIX", "linear")
class AscendW8A8PDMixLinearMethod(AscendW8A8DynamicLinearMethod):
    """Linear method for W8A8 prefill-decode mix.
    
    Uses static W8A8 for KV consumer (decode) and dynamic W8A8 for prefill.
    """

    def __init__(self):
        self.kv_transfer_config = get_current_vllm_config().kv_transfer_config
        super().__init__()

    def apply(self, layer, x, bias=None, tp_rank=0):
        if layer.is_kv_consumer:
            return AscendW8A8LinearMethod.apply(self, layer, x, bias, tp_rank)
        else:
            return AscendW8A8DynamicLinearMethod.apply(self, layer, x, bias,
                                                       tp_rank)

    def get_pertensor_param(self, params_dtype: torch.dtype) -> Dict[str, Any]:
        return AscendW8A8LinearMethod.get_pertensor_param(self, params_dtype)

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        return AscendW8A8LinearMethod.get_perchannel_param(
            self, output_size, params_dtype)

    def process_weights_after_loading(self, layer):
        AscendW8A8LinearMethod.process_weights_after_loading(
            cast(AscendW8A8LinearMethod, self), layer)
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.is_kv_consumer = self.kv_transfer_config is not None and self.kv_transfer_config.is_kv_consumer


@register_scheme("W8A8_MIX", "moe")
class AscendW8A8PDMixFusedMoeMethod(AscendW8A8DynamicFusedMoEMethod):
    """FusedMoE method for W8A8 prefill-decode mix."""

    def __init__(self):
        super().__init__()

    def get_dynamic_quant_param(self, num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = AscendW8A8DynamicFusedMoEMethod.get_dynamic_quant_param(
            self, num_experts, intermediate_size_per_partition, hidden_sizes,
            params_dtype)
        param_dict["w2_deq_scale"] = torch.empty(num_experts,
                                                 hidden_sizes,
                                                 dtype=torch.float32)
        param_dict["w13_deq_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            dtype=torch.float32)
        param_dict["w2_input_offset"] = torch.empty(num_experts,
                                                    1,
                                                    dtype=torch.int8)
        param_dict["w13_input_offset"] = torch.empty(num_experts,
                                                     1,
                                                     dtype=torch.int8)

        return param_dict
