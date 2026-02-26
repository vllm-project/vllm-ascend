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

from __future__ import annotations

import torch
import torch.nn as nn
import torch_npu
from vllm.model_executor.layers.linear import (
    LinearBase,
    QuantizeMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


class AscendUnquantizedLinearMethod310(UnquantizedLinearMethod):
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if "conv1d" not in getattr(layer, "prefix", ""):
            layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, ACL_FORMAT_FRACTAL_NZ)


class AscendLinearBase310(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: object | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        nn.Module.__init__(self)

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.skip_bias_add = skip_bias_add
        self.params_dtype = torch.float16
        self.quant_config = quant_config
        self.prefix = prefix
        self.return_bias = return_bias
        self.disable_tp = disable_tp

        if quant_config is None:
            self.quant_method: QuantizeMethodBase | None = AscendUnquantizedLinearMethod310()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
