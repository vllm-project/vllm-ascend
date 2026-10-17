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

import os
import weakref

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

# Keep the copy bounded to small decode and chunked-prefill collectives. Larger
# collectives keep the in-place path to avoid adding material bandwidth cost.
_AIV_OUT_OF_PLACE_COPY_LIMIT_BYTES = 2 * 1024 * 1024


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
    _instances: weakref.WeakSet = weakref.WeakSet()

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
        self._pending_aiv_outputs: list[torch.Tensor] = []
        self._last_aiv_work = None
        self._instances.add(self)
        self._use_aiv = os.getenv("HCCL_OP_EXPANSION_MODE", "").upper() == "AIV"

    def release_aiv_outputs(self) -> None:
        active_instances = [instance for instance in self._instances if instance._last_aiv_work is not None]
        if not active_instances:
            return

        for instance in active_instances:
            instance._last_aiv_work.wait()
        for device in {instance.device for instance in active_instances}:
            torch.npu.synchronize(device)

        # AIV reads peer storage directly. Local completion alone does not
        # guarantee that a slower peer has finished reading this rank's
        # buffer, so all TP ranks must reach the retirement fence first.
        dist.barrier(group=self.device_group)
        for instance in active_instances:
            instance._pending_aiv_outputs.clear()
            instance._last_aiv_work = None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # vLLM registers its collective custom op as out-of-place, while the
        # default implementation reduces input_ in place and returns the same
        # storage. Small asynchronous AIV collectives can still be consuming
        # that storage when vLLM reuses it. Give those collectives independent
        # output storage and preserve the normal fast path for large messages.
        message_size = input_.numel() * input_.element_size()
        if self._use_aiv and message_size <= _AIV_OUT_OF_PLACE_COPY_LIMIT_BYTES:
            output = input_.clone()
            work = dist.all_reduce(output, group=self.device_group, async_op=True)
            work.wait()
            self._pending_aiv_outputs.append(output)
            self._last_aiv_work = work
            return output
        return super().all_reduce(input_)
