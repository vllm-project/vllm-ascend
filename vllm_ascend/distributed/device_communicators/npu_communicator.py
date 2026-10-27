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


def _is_tp_group(unique_name: str) -> bool:
    """Whether this communicator belongs to the tensor-parallel group.

    ``unique_name`` is ``"<group_name>:<counter>"`` (e.g. ``"tp:0"``,
    ``"ep:0"``, ``"p_tp_0:0"``); only the base name identifies the parallel
    dimension. TP-like variants (``p_tp_*``, ``mlptp``, ``otp``, ``emtp``,
    ``etp``) have different collective semantics, so match ``tp`` exactly.
    """
    return unique_name.split(":")[0] == "tp"


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
        # Optional int8 quantized all-reduce for large bf16 TP collectives,
        # configured through --additional-config (ascend_config). The switch is
        # scoped to the tensor-parallel group: this communicator is also built
        # for dp/ep/mc2/... groups, whose sizes can exceed the supported TP
        # range and whose collectives are out of scope.
        quantized_cfg = get_ascend_config().quantized_all_reduce_config
        self.quantized_ar: QuantizedAllReduce | None = None
        if quantized_cfg.enable and _is_tp_group(self.unique_name) and self.world_size > 1:
            self.quantized_ar = QuantizedAllReduce(
                self.device_group,
                min_numel=quantized_cfg.min_numel,
                world_size=self.world_size,
            )
            logger.info(
                "Quantized all-reduce enabled (min_numel=%d, max_q=%d).",
                self.quantized_ar.min_numel,
                self.quantized_ar.max_q,
            )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.quantized_ar is not None and self.world_size > 1:
            return self.quantized_ar.all_reduce(input_)
        dist.all_reduce(input_, group=self.device_group)
        return input_
