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

import torch
import torch.distributed as dist


class NPUCommunicator:

    def __init__(self, group, unique_name=""):
        self.group = group
        self.unique_name = unique_name
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(self.group)
        self.ranks = dist.get_process_group_ranks(self.group)
        global_rank = dist.get_rank()
        self.rank_in_group = dist.get_group_rank(self.group, global_rank)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1):
        # NOTE: We assume that the input tensor is on the same device across
        # all the ranks.
        # NOTE: `dst` is the local rank of the destination rank.
        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [
                torch.empty_like(input_) for _ in range(self.world_size)
            ]
        else:
            gather_list = None
        # Gather.
        dist.gather(input_, gather_list, dst=self.ranks[dst], group=self.group)
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, input_, group=self.group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (self.world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor
