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
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.device_communicators.quantized_all_reduce import (
    QuantizedAllReduce,
)


class _NpuAll2AllManager:
    """All2All-manager adapter for MC2 and MegaMoe fault tolerance.

    Shares CPU dead-rank state across both backends. MC2 encodes it in
    ``elastic_info``; MegaMoe uses its own device mask bound before capture.
    The public interface mirrors the upstream All2AllManagerBase mask API.
    """

    # MC2 kernels do not detect faults themselves; the mask is written
    # host-side by FT recovery, so a per-step query can never observe one.
    support_fault_tolerance = False

    def __init__(self, ep_world_size: int, device: torch.device | None = None) -> None:
        self._ep_world_size = ep_world_size
        self._device = device
        self._dead: set[int] = set()
        self._num_local_experts: int = 0
        self._mega_moe_buffer = None

        # elastic_info layout: [is_scaling_down, dense ep world size,
        #  shared_expert_rank_num, num_physical_experts] + table1(orig->dense)
        #  + table2(dense->orig). num_physical_experts is derived from the
        #  dead set and num_local_experts on every rebuild.
        size = 4 + 2 * ep_world_size
        self._mc2_elastic_info_host = torch.zeros(size, dtype=torch.int32)
        if device is None:
            device = torch.device("npu", torch.npu.current_device())
        self._mc2_elastic_info = torch.zeros(size, dtype=torch.int32, device=device)

    @property
    def has_mega_moe_buffer(self) -> bool:
        return self._mega_moe_buffer is not None

    @torch.inference_mode()
    def bind_mega_moe_buffer(self, buffer) -> None:
        """Attach MegaMoe's stable rank mask before graph capture.

        EP and MC2 use the same rank order. MegaMoe preserves physical
        expert ids, whereas MC2 uses elastic_info's dense ids.
        """
        # Allocate the mask before graph capture.
        buffer.clean_mask_buffer()
        self._mega_moe_buffer = buffer

    @torch.inference_mode()
    def update_mask(self, rank: int, masked: bool = True) -> None:
        """Update CPU liveness and the active backend's mask state in place."""
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < self._ep_world_size:
            raise ValueError(f"EP rank must be in [0, {self._ep_world_size}), got {rank}.")
        if masked:
            self._dead.add(rank)
        else:
            self._dead.discard(rank)
        if self.has_mega_moe_buffer:
            self._mega_moe_buffer.update_mask_buffer(rank, masked)
        else:
            self._rebuild_mc2_elastic_info()

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

    @torch.inference_mode()
    def clean_buffers(self) -> None:
        """Clear local communication and mask state after device reset.

        The upstream retry flow replays the cumulative dead ranks afterwards.
        Keep all captured tensor storage in place.
        """
        if self.has_mega_moe_buffer:
            self._mega_moe_buffer.get_local_buffer_tensor(torch.uint8).zero_()
            self._mega_moe_buffer.clean_mask_buffer()
        else:
            self._mc2_elastic_info_host.zero_()
            self._mc2_elastic_info.zero_()
        self._dead.clear()

    def get_mc2_elastic_info(self) -> torch.Tensor:
        """The device elastic_info tensor for the next MC2 dispatch/combine."""
        return self._mc2_elastic_info

    def set_num_local_physical_experts(self, num_local_experts: int) -> None:
        """Record the physical expert slots per EP rank."""
        self._num_local_experts = num_local_experts

    def orig_to_dense_rank_table(self) -> torch.Tensor:
        """Map each original EP rank to its densified rank (-1 for dead ranks)."""
        table = torch.full((self._ep_world_size,), -1, dtype=torch.int32)
        alive = sorted(set(range(self._ep_world_size)) - self._dead)
        table[alive] = torch.arange(len(alive), dtype=torch.int32)
        return table

    def _rebuild_mc2_elastic_info(self) -> None:
        """Rebuild MC2 rank mappings in place, preserving captured storage."""

        world_size = self._ep_world_size
        alive = sorted(set(range(world_size)) - self._dead)
        num_physical_experts = len(alive) * self._num_local_experts
        table1 = self.orig_to_dense_rank_table()
        table2 = torch.full((world_size,), -1, dtype=torch.int32)
        table2[: len(alive)] = torch.tensor(alive, dtype=torch.int32)
        self._mc2_elastic_info_host.copy_(
            torch.cat([torch.tensor([1, len(alive), 0, num_physical_experts], dtype=torch.int32), table1, table2])
        )
        self._mc2_elastic_info.copy_(self._mc2_elastic_info_host, non_blocking=True)


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
        # Only the EP group's instance is ever looked up (via the upstream
        # get_ep_all2all_manager()); the rest stay dormant.
        self.all2all_manager = _NpuAll2AllManager(dist.get_world_size(cpu_group), device)
        # Optional int8 quantized all-reduce for large bf16 TP collectives,
        # configured through --additional-config (ascend_config). This
        # communicator is also built for dp/ep/mc2/... groups: the wrapper is
        # created for all of them, but only TP-sized groups take the int8 path
        # (oversized groups keep bf16 instead of failing at start-up).
        quantized_cfg = get_ascend_config().quantized_all_reduce_config
        self.quantized_ar: QuantizedAllReduce | None = None
        if quantized_cfg.enable:
            self.quantized_ar = QuantizedAllReduce(self.device_group, min_numel=quantized_cfg.min_numel)
            if self.quantized_ar.supported:
                # Once per worker process: every group constructs a wrapper, but
                # only the TP-sized ones actually enable the int8 path.
                logger.info_once(
                    "Quantized all-reduce enabled (min_numel=%d, max_q=%d).",
                    self.quantized_ar.min_numel,
                    self.quantized_ar.max_q,
                )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.quantized_ar is not None and self.world_size > 1:
            return self.quantized_ar.all_reduce(input_)
        return super().all_reduce(input_)
