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

from typing import Optional

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def transpose_linear_weights(self, layer: torch.nn.Module) -> None:
    param_data = layer.weight.data
    if param_data.dim() == 2:
        layer.weight.data = param_data.transpose(0, 1)


def linear_without_trans(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if bias is None:
        return torch.matmul(x, layer.weight.data)
    else:
        return torch.addmm(bias, x, layer.weight.data)


UnquantizedLinearMethod.apply = linear_without_trans
UnquantizedLinearMethod.process_weights_after_loading = transpose_linear_weights