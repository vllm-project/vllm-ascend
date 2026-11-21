# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense BF16 checkpoint loaders for Ascend's compressed-tensors MX schemes.

Layerwise reload records these dense parameters before postprocessing and
recreates them on subsequent updates. The ordinary Ascend scheme remains
responsible for the inference layout and kernels.
"""

import torch
import torch_npu
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from vllm_ascend.quantization.methods.base import AscendLinearScheme, AscendMoEScheme
from vllm_ascend.quantization.quant_type import QuantType


def _validate_dense(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    if dtype != torch.bfloat16:
        raise ValueError("Online MX weight quantization requires BF16 checkpoint weights")
    # Inference kernels pair two 32-element scale groups.
    if shape[-1] % 64:
        raise ValueError("Online MX weight quantization requires local reduction dimensions divisible by 64")


def _quantize(layer: torch.nn.Module, names: tuple[str, ...], scheme: AscendLinearScheme | AscendMoEScheme) -> None:
    packed = scheme.quant_type == QuantType.W4A8MXFP
    destination = torch_npu.float4_e2m1fn_x2 if packed else torch.float8_e4m3fn
    for name in names:
        param = getattr(layer, name)
        _validate_dense(tuple(param.shape), param.dtype)
        # Flatten experts to rows so both linear and MoE use the same MX axis.
        shape = tuple(param.shape)
        weight, scale = torch_npu.npu_dynamic_mx_quant(
            param.detach().reshape(-1, shape[-1]).contiguous(), dst_type=destination
        )
        if packed:
            weight = weight.view(torch.uint8)
        param.data = weight.reshape(*shape[:-1], -1)
        scale = scale.view(torch.uint8).reshape(*shape[:-1], shape[-1] // 32)
        layer.register_parameter(name + "_scale", torch.nn.Parameter(scale, requires_grad=False))
    # The native schemes guard against processing an already transformed layer.
    # Dense reload parameters above have just replaced that representation.
    layer._mxfp8_transformed = False
    layer._mxfp4_transformed = False
    scheme.process_weights_after_loading(layer)


class AscendOnlineMXLinearMethod(AscendLinearMethod):
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        shape = (sum(output_partition_sizes), input_size_per_partition)
        _validate_dense(shape, params_dtype)
        param = torch.nn.Parameter(torch.empty(shape, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(param, {"input_dim": 1, "output_dim": 0, **extra_weight_attrs})
        layer.register_parameter("weight", param)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _quantize(layer, ("weight",), self.quant_method)


class AscendOnlineMXMoEMethod(AscendFusedMoEMethod):
    @property
    def supports_eplb(self) -> bool:
        return False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if self.quant_method.use_expert_weight_list:
            raise ValueError("Online MX weight quantization does not support EPLB")
        shapes = {
            "w13_weight": (num_experts, 2 * intermediate_size_per_partition, hidden_size),
            "w2_weight": (num_experts, hidden_size, intermediate_size_per_partition),
        }
        for name, shape in shapes.items():
            _validate_dense(shape, params_dtype)
            param = torch.nn.Parameter(torch.empty(shape, dtype=params_dtype), requires_grad=False)
            set_weight_attrs(param, extra_weight_attrs)
            layer.register_parameter(name, param)

    def get_computed_params(self) -> set[str]:
        return {"w13_weight_scale", "w2_weight_scale"}

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _quantize(layer, ("w13_weight", "w2_weight"), self.quant_method)
