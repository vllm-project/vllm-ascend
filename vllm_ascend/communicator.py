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
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.base_device_communicator import \
    DeviceCommunicatorBase


class NPUCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        # init device according to rank
        self.device = torch.npu.current_device()

    def all_to_all(self,
                   input_: torch.Tensor,
                   scatter_dim: int = 0,
                   gather_dim: int = -1) -> torch.Tensor:

        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()
        if scatter_dim == gather_dim:
            return input_

        input_list = [
            t.contiguous()
            for t in torch.tensor_split(input_, self.world_size, scatter_dim)
        ]
        output_list = [
            torch.empty_like(input_list[0]) for _ in range(self.world_size)
        ]

        dist.all_to_all(output_list, input_list, group=self.device_group)
        output_tensor = torch.cat(output_list, dim=gather_dim).contiguous()
        return output_tensor

    def reduce_scatter(self,
                       input_: torch.Tensor,
                       scatter_dim: int = 0) -> torch.Tensor:

        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if scatter_dim != 0:
            input_ = torch.transpose(input_, 0, scatter_dim)
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] // self.world_size
        output_tensor = torch.empty(dim_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        dist.reduce_scatter_tensor(output_tensor,
                                   input_.contiguous(),
                                   group=self.device_group)
        if scatter_dim != 0:
            output_tensor = torch.transpose(output_tensor, 0,
                                            scatter_dim).contiguous()
        return output_tensor
