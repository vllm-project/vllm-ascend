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

    def clean_buffers(self) -> None:
        """No-op: NPU registers no RDMA buffers or all2all mask state."""


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
        self.all2all_manager = _NpuAll2AllManager()
