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
from vllm_ascend.distributed.device_communicators.hccl_lifecycle import HcclSleepAdapter
from vllm_ascend.distributed.device_communicators.mxfp8_all_reduce import Mxfp8AllReduce
from vllm_ascend.distributed.device_communicators.quantized_all_reduce import (
    QuantizedAllReduce,
)


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
        sleep_adapter: HcclSleepAdapter | None = None,
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
        self._sleep_adapter = sleep_adapter
        self._destroyed = False
        self.quantized_ar: QuantizedAllReduce | Mxfp8AllReduce | None = None
        self._init_quantized_all_reduce()

    def _init_quantized_all_reduce(self) -> None:
        # Select the configured backend; defaults are MXFP8 on A5, INT8 on A2.
        # Both wrappers use payload/shape gates and the existing process group.
        quantized_cfg = get_ascend_config().quantized_all_reduce_config
        self.quantized_ar = None
        if not quantized_cfg.enable or self.device_group is None:
            return
        if quantized_cfg.quant_dtype == "int8":
            self.quantized_ar = QuantizedAllReduce(
                self.device_group, min_numel=quantized_cfg.min_numel, group_name=self.unique_name
            )
            if self.quantized_ar.supported:
                logger.info_once(
                    "INT8 quantized all-reduce enabled (group=%s, min_numel=%d, max_q=%d).",
                    self.unique_name,
                    self.quantized_ar.min_numel,
                    self.quantized_ar.max_q,
                )
        else:
            self.quantized_ar = Mxfp8AllReduce(
                self.device_group,
                min_numel=quantized_cfg.min_numel,
                group_name=self.unique_name,
            )
            if self.quantized_ar.supported:
                logger.info_once(
                    "MXFP8 AllToAll quantized all-reduce enabled "
                    "(group=%s, min_numel=%d, dequant_dtype=bf16, sum_output_dtype=bf16).",
                    self.unique_name,
                    self.quantized_ar.min_numel,
                )

    def bind_device_group(self, device_group: dist.ProcessGroup | None) -> None:
        """Rebind after HCCL sleep restore; refresh PG-backed quantized AR."""
        self.device_group = device_group
        self._init_quantized_all_reduce()

    def suspend(self) -> None:
        """Release reclaimable HCCL memory using the coordinator's PG owner.

        The current backend is destroy/restore, not native HcclCommSuspend:
        native suspension has not demonstrated HBM reclamation and reliable
        multi-rank recovery. The NPU worker invalidates ACL graphs first when
        this cycle will rebuild HCCL.
        """
        if self._destroyed:
            raise RuntimeError("Cannot suspend a destroyed NPU communicator.")
        if self._sleep_adapter is None:
            raise RuntimeError("This NPU communicator has no HCCL sleep adapter.")
        self._sleep_adapter.suspend()

    def resume(self) -> None:
        """Rebind restored PG resources without replacing this communicator."""
        if self._destroyed:
            raise RuntimeError("Cannot resume a destroyed NPU communicator.")
        if self._sleep_adapter is None:
            raise RuntimeError("This NPU communicator has no HCCL sleep adapter.")
        self._sleep_adapter.resume()

    def destroy(self) -> None:
        # PG ownership stays with GroupCoordinator/HcclPgRegistry. Final
        # coordinator teardown closes its adapter and releases any held PGs.
        self._destroyed = True
        self.quantized_ar = None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.quantized_ar is not None and self.world_size > 1:
            return self.quantized_ar.all_reduce(input_)
        return super().all_reduce(input_)
