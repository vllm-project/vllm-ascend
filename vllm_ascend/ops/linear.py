#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch
import torch_npu

from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def process_weights_after_loading(
    self,
    layer: torch.nn.Module
):
    layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)


UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading