#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from typing import Any, Union

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.utils import AWQ_QUANTIZATION_METHOD

from .method_adapters import AscendFusedMoEMethod
from .methods import get_scheme_class
from .methods.w4a16_awq import AscendW4A16AWQLinearMethod


@register_quantization_config(AWQ_QUANTIZATION_METHOD)
class AWQConfig(QuantizationConfig):
    """AWQ quantization config for Ascend NPU.

    Replaces vLLM's native AWQ config to route linear and MoE layers through
    Ascend-specific scheme implementations (AscendW4A16AWQLinearMethod,
    AscendW4A16AWQFusedMoEMethod) .
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
        quant_config: dict[str, Any] | None = None,
    ):
        self.quant_description = quant_config if quant_config is not None else {}
        super().__init__()

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                f"Currently, only 4-bit weight quantization is supported for AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def get_name(self) -> str:
        return AWQ_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError("Ascend hardware does not support 'get_min_capability' feature.")

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert, config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return AscendUnquantizedLinearMethod()
            return AscendW4A16AWQLinearMethod(self)

        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                skip_with_substr=True,
            ):
                return AscendUnquantizedFusedMoEMethod(layer.moe_config)
            scheme_cls = get_scheme_class("W4A16_AWQ", "moe")
            if scheme_cls is None:
                raise NotImplementedError(f"W4A16_AWQ moe scheme not found for layer {prefix}")
            return AscendFusedMoEMethod(scheme_cls(self), layer.moe_config)

        return None
