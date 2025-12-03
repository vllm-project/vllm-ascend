from typing import Any, Dict, cast

import torch
from vllm.config import get_current_vllm_config

from .w8a8 import AscendW8A8LinearMethod
from .w8a8_dynamic import (AscendW8A8DynamicFusedMoEMethod,
                           AscendW8A8DynamicLinearMethod)


class AscendW8A8PDMixLinearMethod(AscendW8A8DynamicLinearMethod):

    def __init__(self):
        self.kv_transfer_config = get_current_vllm_config().kv_transfer_config
        super().__init__()

    @staticmethod
    def apply(layer, x, bias=None, tp_rank=0):
        if layer.is_kv_consumer:
            return AscendW8A8LinearMethod.apply(layer, x, bias, tp_rank)
        else:
            return AscendW8A8DynamicLinearMethod.apply(layer, x, bias, tp_rank)

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return AscendW8A8LinearMethod.get_pertensor_param(params_dtype)

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        return AscendW8A8LinearMethod.get_perchannel_param(
            output_size, params_dtype)

    def process_weights_after_loading(self, layer):
        AscendW8A8LinearMethod.process_weights_after_loading(
            cast(AscendW8A8LinearMethod, self), layer)
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.is_kv_consumer = self.kv_transfer_config is not None and self.kv_transfer_config.is_kv_consumer


class AscendW8A8PDMixFusedMoeMethod(AscendW8A8DynamicFusedMoEMethod):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_dynamic_quant_param(num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = AscendW8A8DynamicFusedMoEMethod.get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_sizes,
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
