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

# elastic_info tensor layout consumed by the MC2 dispatch/combine operators:
# [is_scaling_down, dense ep world size, shared_expert_rank_num,
#  num_physical_experts] + table1(orig->dense, -1 for dead) + table2(dense->orig).
_ELASTIC_INFO_HEADER_SIZE = 4
_ELASTIC_INFO_RANK_TABLE_NUM = 2


class _NpuAll2AllManager:
    """All2All-manager adapter for MC2 fault tolerance.

    Owns the dead-rank mask together with the `elastic_info` tensor the mask
    is encoded into: the MC2 dispatch/combine operators take `elastic_info`
    fresh on every call, so this tensor doubles as the mask (there is no
    kernel-side mask buffer like DeepEP/nixl-ep). The public interface mirrors
    the upstream All2AllManagerBase mask API so the upstream FT sentinel
    drives it unchanged; future Ascend operators with FT support are expected
    to follow the same shape.
    """

    # Unlike DeepEP/nixl-ep, the MC2 kernels neither detect faults nor set
    # the mask themselves on timeout — a dead peer surfaces as an aborted op
    # raising out of the forward, and the mask is only ever written host-side
    # by FT recovery (scale_down, plus the retry mask replay). A per-step
    # query_fault() could therefore never observe anything; reporting False
    # keeps the upstream runners from paying for that query every step.
    support_fault_tolerance = False

    def __init__(self, ep_world_size: int, device: torch.device | None = None) -> None:
        self._ep_world_size = ep_world_size
        self._device = device
        self._dead: set[int] = set()
        self._num_physical_experts: int = 0

        size = _ELASTIC_INFO_HEADER_SIZE + _ELASTIC_INFO_RANK_TABLE_NUM * ep_world_size
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
        # NPU counterpart of the upstream per-step fault check. Unlike
        # DeepEP/nixl-ep there is no in-kernel timeout that flips the mask,
        # so a fault can never be observed this way — always report no fault.
        # (Faults surface as aborted HCCL ops raising out of execute_model;
        # see support_fault_tolerance.)
        return torch.tensor(False)

    def clean_buffers(self) -> None:
        """No-op, kept for the upstream retry flow which calls it
        unconditionally.

        Unlike DeepEP/nixl-ep there is no kernel-side mask buffer or RDMA
        state to clean: the elastic_info tensor is passed fresh to every MC2
        call, so the mask itself is intentionally left untouched (it survives
        across recovery rounds via the replayed cumulative dead set).
        """

    def get_elastic_info(self) -> torch.Tensor:
        """The device elastic_info tensor for the next MC2 dispatch/combine."""
        return self._elastic_info

    def set_num_physical_experts(self, num_physical_experts: int) -> None:
        """Shrink the expert-space width after scale-down and rebuild."""
        self._num_physical_experts = num_physical_experts
        self._rebuild_elastic_info()

    def to_densified_rank_table(self) -> torch.Tensor:
        """Convert the scale-down view (original EP ranks + dead set) into the
        kernel's view: original EP rank -> densified rank (-1 = dead)."""
        table = torch.full((self._ep_world_size,), -1, dtype=torch.int32)
        alive = sorted(set(range(self._ep_world_size)) - self._dead)
        for dense_rank, orig_rank in enumerate(alive):
            table[orig_rank] = dense_rank
        return table

    def _rebuild_elastic_info(self) -> None:
        """Rebuild elastic_info from the dead set into the existing device
        tensor (never reallocates, so captured graphs stay valid)."""
        if not self._dead or self._num_physical_experts <= 0:
            # flag=0 -> normal dispatch; transient until redistribution sets
            # the new expert width (no forward runs in between).
            self._elastic_info_host.zero_()
        else:
            world_size = self._ep_world_size
            alive = sorted(set(range(world_size)) - self._dead)
            table1 = torch.full((world_size,), -1, dtype=torch.int32)
            table1[alive] = torch.arange(len(alive), dtype=torch.int32)
            table2 = torch.full((world_size,), -1, dtype=torch.int32)
            table2[: len(alive)] = torch.tensor(alive, dtype=torch.int32)
            self._elastic_info_host.copy_(
                torch.cat(
                    [torch.tensor([1, len(alive), 0, self._num_physical_experts], dtype=torch.int32), table1, table2]
                )
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
