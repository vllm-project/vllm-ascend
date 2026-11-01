#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""310P compressed-tensors configuration, extending treason258's PR #11427."""

import torch
import torch_npu
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.quantization import register_quantization_config

from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.methods import AscendLinearScheme, AscendMoEScheme
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD

from .methods.registry import get_scheme_class
from .methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod310


class AscendCompressedTensorsW8A8DynamicLinearMethod310(AscendW8A8DynamicLinearMethod310):
    """Use contiguous NK weights without changing the ModelSlim NZ scheme."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if x.dtype != torch.float16:
            raise ValueError("310P compressed-tensors W8A8 requires float16 activations.")
        original_shape = x.shape
        output = torch_npu.npu_quant_matmul_dequant(
            x.reshape(-1, original_shape[-1]).contiguous(),
            layer.weight.data,
            layer.weight_scale,
            quant_mode="pertoken",
        )
        # CANN's 310P QuantMatmulDequant requires biasOptional to be null.
        if bias is not None:
            output = output + bias
        return output.reshape(*original_shape[:-1], output.shape[-1])

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = layer.weight.data.contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()


@register_quantization_config(COMPRESSED_TENSORS_METHOD)
class AscendCompressedTensorsConfig310(AscendCompressedTensorsConfig):
    """310P override for dense INT8 per-token compressed-tensors checkpoints."""

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    def _create_scheme_for_layer_type(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        format: str | None,
        layer_type: str,
    ) -> AscendLinearScheme | AscendMoEScheme:
        quant_type = self._detect_quant_type(weight_quant, input_quant, format)
        quant_format = format if format is not None else self.quant_format
        scheme_cls = get_scheme_class(quant_type, layer_type)
        if (
            scheme_cls is None
            or quant_type != "W8A8_DYNAMIC"
            or layer_type != "linear"
            or quant_format != "int-quantized"
        ):
            raise NotImplementedError(
                f"No 310P compressed-tensors compatible scheme was found for "
                f"quant_type={quant_type}, layer_type={layer_type}, format={quant_format}. "
                "Only dense INT8 W8A8 per-token dynamic quantization is supported."
            )
        return AscendCompressedTensorsW8A8DynamicLinearMethod310()
