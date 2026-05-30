#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# Patch: Mark DeepSeekMTP as PP-compatible so that config validation during
# API server startup (verify_with_parallel_config) passes before worker patches
# are loaded.
#

import torch
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.sequence import IntermediateTensors

DeepSeekMTP.supports_pp = True


def _mtp_make_empty_intermediate_tensors(
    self,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> IntermediateTensors:
    return IntermediateTensors({})


DeepSeekMTP.make_empty_intermediate_tensors = _mtp_make_empty_intermediate_tensors
