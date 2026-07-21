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
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
from vllm.distributed.utils import StatelessProcessGroup


class _NpuAll2AllManager:
    """No-op all2all_manager for NPU. Used by vLLM main's fault-tolerance
    check (data_parallel_size > 1 and is_moe); NPU does not register a real
    one because it uses mc2 / all_gather for MoE communication.
    """

    @property
    def support_fault_tolerance(self) -> bool:
        return False

    def query_fault(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    def query_active_mask(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
        )
        # TODO(hz): Refer to CudaCommunicator's implementation to integrate PyHcclCommunicator
        # init device according to rank
        self.device = torch.npu.current_device()

        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

        self.pyhccl_comm: PyHcclCommunicator | None = None
        if self.world_size > 1 and tcp_store_group is not None:
            self.pyhccl_comm = PyHcclCommunicator(group=tcp_store_group, device=self.device)

        # For compatibility (mainly for reusing graph capturing code in vllm),
        # init custom all-reduce implementation interface as in CUDACommunicator.
        self.ca_comm = None
        self.all2all_manager = _NpuAll2AllManager()

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]
            output_list = []
            tensor_shape_base = input_list[self.rank].size()
            for i in range(self.world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[i]
                output_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))

        else:
            input_list = [t.contiguous() for t in torch.tensor_split(input_, self.world_size, scatter_dim)]
            output_list = [torch.empty_like(input_list[i]) for i in range(self.world_size)]

        dist.all_to_all(output_list, input_list, group=self.device_group)
        output_tensor = torch.cat(output_list, dim=gather_dim).contiguous()
        return output_tensor

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.pyhccl_comm is not None:
            if dim < 0:
                # Convert negative dim to positive.
                dim += input_.dim()
            input_size = input_.size()
            # NOTE: we have to use concat-style all-gather here,
            # stack-style all-gather has compatibility issues with
            # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
            output_size = (input_size[0] * self.world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
            # All-gather.
            output_tensor = self.pyhccl_comm.all_gather(input_, output_tensor)
            # Reshape
            output_tensor = output_tensor.reshape((self.world_size,) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1 :]
            )
            return output_tensor
        else:
            return super().all_gather(input_, dim)

    def destroy(self):
        if self.pyhccl_comm is not None:
            self.pyhccl_comm.destroy()
            self.pyhccl_comm = None

    def batch_isend_irecv(self, p2p_ops: list):
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is not None and not pyhccl_comm.disabled:
            pyhccl_comm.batch_isend_irecv(p2p_ops)
        else:
            raise ValueError("No PyHccl communicator found")
