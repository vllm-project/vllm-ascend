from typing import Any, Dict

import torch
import torch_npu

from vllm_ascend.ascend_config import PDScenarioState, get_ascend_config
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_enable_nz

from .w8a8 import AscendW8A8LinearMethod
from .w8a8_dynamic import (AscendW8A8DynamicFusedMoEMethod,
                           AscendW8A8DynamicLinearMethod)


class AscendW8A8PDMixLinearMethod(AscendW8A8DynamicLinearMethod):

    def __init__(self):
        super().__init__()

    @staticmethod
    def apply(layer, x, bias=None, tp_rank=0):
        if get_ascend_config().pd_scenario_state == PDScenarioState.DNode:
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
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_scale_reciprocal = torch.nn.Parameter(
            1.0 / layer.aclnn_input_scale.data,
            requires_grad=False)
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor),
            requires_grad=False).to(layer.aclnn_input_scale.dtype)
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        if is_enable_nz():
            layer.weight.data = torch_npu.npu_format_cast(
                layer.weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)


class AscendW8A8PDMixFusedMoeMethod(AscendW8A8DynamicFusedMoEMethod):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_dynamic_quant_param(num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=params_dtype)
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=params_dtype)
        param_dict["w2_weight_scale"] = torch.empty(num_experts,
                                                    hidden_sizes,
                                                    1,
                                                    dtype=params_dtype)
        param_dict["w2_weight_offset"] = torch.empty(num_experts,
                                                     hidden_sizes,
                                                     1,
                                                     dtype=params_dtype)
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
