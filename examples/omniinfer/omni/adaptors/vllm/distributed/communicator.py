# SPDX-License-Identifier: Apache-2.0
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

from typing import List, Optional

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import \
    DeviceCommunicatorBase


class NPUCommunicator(DeviceCommunicatorBase):
    def all_to_all(
            self,
            input_: torch.Tensor,
            scatter_dim: int = 0,
            gather_dim: int = -1,
            scatter_sizes: Optional[List[int]] = None,
            gather_sizes: Optional[List[int]] = None
    ) -> torch.Tensor:
        scatter_dim = self._normalize_dim(scatter_dim, input_.dim())
        gather_dim = self._normalize_dim(gather_dim, input_.dim())
        self._validate_inputs(input_, scatter_dim, gather_sizes, scatter_sizes)
        input_list = self._prepare_input_list(input_, scatter_dim, scatter_sizes)
        output_list = self._prepare_output_list(input_list, gather_dim, gather_sizes)
        dist.all_to_all(output_list, input_list, group=self.device_group)
        return torch.cat(output_list, dim=gather_dim).contiguous()

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> Optional[torch.Tensor]:
        output_tensor = self.all_gather(input_, dim)
        if self.rank_in_group == dst:
            return output_tensor
        else:
            return None

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        input_size = tuple(input_.size())
        output_tensor = torch.empty(
            (input_size[0] // self.world_size,) + input_size[1:],
            dtype=input_.dtype,
            device=input_.device,
        )
        torch.distributed.reduce_scatter_tensor(
            output_tensor, input_, group=self.device_group
        )
        return output_tensor

    def _normalize_dim(self, dim: int, tensor_dim: int) -> int:
        if dim < -tensor_dim or dim >= tensor_dim:
            raise ValueError(f"Dimension {dim} is out of bounds for tensor with {tensor_dim} dimensions")
        return dim if dim >= 0 else dim + tensor_dim

    def _validate_inputs(
            self,
            input_: torch.Tensor,
            scatter_dim: int,
            gather_sizes: Optional[List[int]],
            scatter_sizes: Optional[List[int]]
        ) -> None:
        if scatter_sizes is not None and gather_sizes is not None:
            if len(scatter_sizes) != self.world_size or len(gather_sizes) != self.world_size:
                raise ValueError(
                    f"scatter_sizes and gather_sizes must have length {self.world_size}, "
                    f"got {len(scatter_sizes)} and {len(gather_sizes)}"
                )
            if sum(scatter_sizes) != input_.size(scatter_dim):
                raise ValueError(
                    f"Sum of scatter_sizes ({sum(scatter_sizes)}) does not match "
                    f"input size at scatter_dim ({input_.size(scatter_dim)})"
                )
        elif scatter_sizes is not None or gather_sizes is not None:
            raise ValueError("Both scatter_sizes and gather_sizes must be provided or neither")

    def _prepare_input_list(
            self,
            input_: torch.Tensor,
            scatter_dim: int,
            scatter_sizes: Optional[List[int]]
    ) -> List[torch.Tensor]:
        if scatter_sizes is not None:
            return [
                t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)
            ]
        return [
            t.contiguous()
            for t in torch.tensor_split(input_, self.world_size, scatter_dim)
        ]

    def _prepare_output_list(
        self,
        input_list: List[torch.Tensor],
        gather_dim: int,
        gather_sizes: Optional[List[int]],
    ) -> List[torch.Tensor]:
        if gather_sizes is not None:
            tensor_shape_base = input_list[self.rank].size()
            output_list = []
            for size in gather_sizes:
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = size
                output_list.append(
                    torch.empty(
                        tensor_shape,
                        dtype=input_list[0].dtype,
                        device=input_list[0].device,
                    )
                )
        else:
            output_list = [torch.empty_like(chunk) for chunk in input_list]
        return output_list

    def all_reduce_async(self, input_: torch.Tensor) -> torch.Tensor:
        handle = dist.all_reduce(input_, group=self.device_group, async_op=True)
        return input_, handle

    def all_gather_async(self, input_: torch.Tensor, dim: int = -1):
        assert dim == 0, "all_gather_async only supports dim=0"
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
        handle = dist.all_gather_into_tensor(output_tensor,
                                             input_,
                                             group=self.device_group,
                                             async_op=True)
        return output_tensor, handle

    def reduce_scatter_async(self, input_: torch.Tensor):
        input_size = tuple(input_.size())
        output_tensor = torch.empty(
            (input_size[0] // self.world_size,) + input_size[1:],
            dtype=input_.dtype,
            device=input_.device,
        )
        handle = torch.distributed.reduce_scatter_tensor(
            output_tensor, input_, group=self.device_group, async_op=True
        )
        return output_tensor, handle

    def all_gather_v(self, output_list_: List[torch.Tensor], input_:torch.Tensor) -> None:
        torch.distributed.all_gather(output_list_, input_, group=self.device_group)

    def reduce_scatter_v(self, output_:torch.Tensor, input_list_:List[torch.Tensor]) -> None:
        torch.distributed.reduce_scatter(output_, input_list_, group=self.device_group)