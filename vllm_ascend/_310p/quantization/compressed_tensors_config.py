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
"""310P compressed-tensors quantization configuration."""

from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.quantization import register_quantization_config

from vllm_ascend.quantization.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.methods import AscendLinearScheme, AscendMoEScheme
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD


@register_quantization_config(COMPRESSED_TENSORS_METHOD)
class AscendCompressedTensorsConfig310(AscendCompressedTensorsConfig):
    """310P override for LLM-Compressor compressed-tensors quantization."""

    def _create_scheme_for_layer_type(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        format: str | None,
        layer_type: str,
    ) -> AscendLinearScheme | AscendMoEScheme:
        import vllm_ascend._310p.quantization.methods  # noqa: F401
        from vllm_ascend._310p.quantization.methods.registry import get_scheme_class

        quant_type = self._detect_quant_type(weight_quant, input_quant, format)
        scheme_cls = get_scheme_class(quant_type, layer_type)
        if scheme_cls is None:
            raise NotImplementedError(
                f"No 310P compressed-tensors compatible scheme was found for "
                f"quant_type={quant_type}, layer_type={layer_type}."
            )

        return scheme_cls()
