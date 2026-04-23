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

from typing import Any

import torch

from vllm_ascend.quantization.methods.base import AscendLinearScheme


class AscendW8A8Linear310pScheme(AscendLinearScheme):
    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.float16,
    ) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}

    def get_pertensor_param(self, params_dtype: torch.dtype) -> dict[str, Any]:
        return {
            "input_scale": torch.empty(1, dtype=params_dtype),
            "input_offset": torch.empty(1, dtype=torch.int8),
        }

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {
            "quant_bias": torch.empty(output_size, dtype=torch.int32),
            "deq_scale": torch.empty(output_size, dtype=torch.int64),
        }
