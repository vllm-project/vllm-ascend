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
    """All2All-manager adapter for MC2 fault tolerance.

    Owns the dead-rank mask, encoded into the ``elastic_info`` tensor consumed
    by the MC2 dispatch/combine operators. The public interface mirrors the
    upstream All2AllManagerBase mask API.
    """

    # MC2 kernels do not detect faults themselves; the mask is written
    # host-side by FT recovery, so a per-step query can never observe one.
    support_fault_tolerance = False

    def __init__(self, ep_world_size: int, device: torch.device | None = None) -> None:
        self._ep_world_size = ep_world_size
        self._device = device
        self._dead: set[int] = set()
        self._num_local_experts: int = 0

        # elastic_info layout: [is_scaling_down, dense ep world size,
        #  shared_expert_rank_num, num_physical_experts] + table1(orig->dense)
        #  + table2(dense->orig). num_physical_experts is derived from the
        #  dead set and num_local_experts on every rebuild.
        size = 4 + 2 * ep_world_size
        self._elastic_info_host = torch.zeros(size, dtype=torch.int32)
        if device is None:
            device = torch.device("npu", torch.npu.current_device())
        self._elastic_info = torch.zeros(size, dtype=torch.int32, device=device)

    def update_mask(self, rank: int, masked: bool = True) -> None:
        """Mark an EP rank dead/alive and rebuild elastic_info in place."""
        if masked:
            self._dead.add(rank)
        else:
            self._dead.discard(rank)
        self._rebuild_elastic_info()

    def query_active_mask(self) -> torch.Tensor:
        """Per-EP-rank mask (1=dead, 0=live) as a CPU tensor, matching the
        upstream mask-buffer convention.

        Built on CPU on purpose: this is called while a fault is being
        probed, when the NPU may be hung — any device op would fail.
        """
        mask = torch.zeros(self._ep_world_size, dtype=torch.int32)
        for rank in self._dead:
            mask[rank] = 1
        return mask

    def query_fault(self) -> torch.Tensor:
        # MC2 has no in-kernel fault detection; faults surface as aborted ops.
        return torch.tensor(False)

    def clean_buffers(self) -> None:
        """No-op, kept for the upstream retry flow which calls it unconditionally."""

    def get_elastic_info(self) -> torch.Tensor:
        """The device elastic_info tensor for the next MC2 dispatch/combine."""
        return self._elastic_info

    def set_num_local_physical_experts(self, num_local_experts: int) -> None:
        """Record the physical expert slots per EP rank."""
        self._num_local_experts = num_local_experts

    def _rebuild_elastic_info(self) -> None:
        """Rebuild elastic_info from the dead set into the existing device
        tensor (never reallocates, so captured graphs stay valid)."""

        world_size = self._ep_world_size
        alive = sorted(set(range(world_size)) - self._dead)
        num_physical_experts = len(alive) * self._num_local_experts
        table1 = torch.full((world_size,), -1, dtype=torch.int32)
        table1[alive] = torch.arange(len(alive), dtype=torch.int32)
        table2 = torch.full((world_size,), -1, dtype=torch.int32)
        table2[: len(alive)] = torch.tensor(alive, dtype=torch.int32)
        self._elastic_info_host.copy_(
            torch.cat([torch.tensor([1, len(alive), 0, num_physical_experts], dtype=torch.int32), table1, table2])
        )
        self._elastic_info.copy_(self._elastic_info_host, non_blocking=True)


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
        # Only the EP group's instance is ever looked up (via the upstream
        # get_ep_all2all_manager()); the rest stay dormant.
        self.all2all_manager = _NpuAll2AllManager(dist.get_world_size(cpu_group), device)
