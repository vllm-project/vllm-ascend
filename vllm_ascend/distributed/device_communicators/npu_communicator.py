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
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            use_all2all=use_all2all,
        )
        self.device = torch.npu.current_device()
        self.ca_comm = None
        # vLLM #53576 reads this CUDA-only communicator during graph capture.
        # Keep the shared coordinator protocol available without enabling the
        # FlashInfer PCIe IPC backend on NPU.
        self.fi_pcie_ipc_ar_comm = None
        self.all2all_manager = _NpuAll2AllManager()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Gather rows in group-rank order, including uneven or empty shards.

        Without sizes, all ranks must have equal row counts. Otherwise, sizes
        applies to every tensor in the list and must agree across ranks.
        Shapes, sizes and tensor-list order must stay fixed during graph replay.
        """
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        if sizes is not None:
            if len(sizes) != self.world_size:
                raise ValueError("all_gatherv sizes must match group size")
            if any(size < 0 for size in sizes):
                raise ValueError("all_gatherv sizes must be non-negative")

        def gather(input_: torch.Tensor) -> torch.Tensor:
            shard_sizes = sizes if sizes is not None else [input_.shape[0]] * self.world_size
            if input_.shape[0] != shard_sizes[self.rank_in_group]:
                raise ValueError("all_gatherv local rows do not match sizes")
            output = torch.empty((sum(shard_sizes), *input_.shape[1:]), dtype=input_.dtype, device=input_.device)
            # An all-empty group has no data to exchange. Every rank makes
            # this decision from the same sizes, not its own local row count.
            if output.numel() == 0:
                return output
            dist.all_gather(list(output.split(shard_sizes)), input_.contiguous(), group=self.device_group)
            return output

        if isinstance(input_, torch.Tensor):
            return gather(input_)
        return [gather(tensor) for tensor in input_]
