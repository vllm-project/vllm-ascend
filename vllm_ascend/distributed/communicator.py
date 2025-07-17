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
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn
from vllm.distributed.device_communicators.base_device_communicator import \
    DeviceCommunicatorBase
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.distributed.communication_op import \
    data_parallel_reduce_scatter
from vllm_ascend.utils import dispose_tensor


class NPUCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: dist.ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[dist.ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        # TODO(hz): Refer to CudaCommunicator's implementation to integrate PyHcclCommunicator
        # init device according to rank
        self.device = torch.npu.current_device()

    def all_to_all(self,
                   input_: torch.Tensor,
                   scatter_dim: int = 0,
                   gather_dim: int = -1,
                   scatter_sizes: Optional[List[int]] = None,
                   gather_sizes: Optional[List[int]] = None) -> torch.Tensor:

        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [
                t.contiguous()
                for t in torch.split(input_, scatter_sizes, scatter_dim)
            ]
            output_list = []
            tensor_shape_base = input_list[self.rank].size()
            for i in range(self.world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[i]
                output_list.append(
                    torch.empty(tensor_shape,
                                dtype=input_.dtype,
                                device=input_.device))

        else:
            input_list = [
                t.contiguous() for t in torch.tensor_split(
                    input_, self.world_size, scatter_dim)
            ]
            output_list = [
                torch.empty_like(input_list[i]) for i in range(self.world_size)
            ]

        dist.all_to_all(output_list, input_list, group=self.device_group)
        output_tensor = torch.cat(output_list, dim=gather_dim).contiguous()
        return output_tensor

    def dispatch(
            self, hidden_states: torch.Tensor,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """
        num_tokens, _ = hidden_states.shape
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None:
            max_num_tokens_across_dp = attn_metadata.max_num_tokens_across_dp
            if num_tokens < max_num_tokens_across_dp:
                hidden_states = nn.functional.pad(
                    hidden_states,
                    (0, 0, 0, max_num_tokens_across_dp - num_tokens))
                router_logits = nn.functional.pad(
                    router_logits,
                    (0, 0, 0, max_num_tokens_across_dp - num_tokens))
        hidden_states = get_dp_group().all_gather(hidden_states, 0)
        router_logits = get_dp_group().all_gather(router_logits, 0)

        return hidden_states, router_logits

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        num_tokens, _ = hidden_states.shape
        final_hidden_states = data_parallel_reduce_scatter(hidden_states,
                                                           dim=0)
        final_hidden_states = final_hidden_states[:num_tokens]
        dispose_tensor(hidden_states)
        return hidden_states
