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

from typing import Optional, Tuple, Union

import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              split_tensor_along_first_dim)

def forward_oot(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    is_fc3=False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    import torch_npu

    if is_fc3:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if residual is not None:
            residual_split = split_tensor_along_first_dim(residual, num_partitions=tp_size)[tp_rank].contiguous()

    if residual is not None:
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual_split if is_fc3 else residual, self.weight,
                                                    self.variance_epsilon)
        return x, residual

    x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
    return x


RMSNorm.forward_oot = forward_oot
