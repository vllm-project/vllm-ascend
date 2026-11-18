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
    max_sequence_collective_peer_bytes: int | None = None

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

    def _sequence_chunk_size(self, input_: torch.Tensor, dim: int, shards: int = 1) -> int | None:
        limit = self.max_sequence_collective_peer_bytes
        if limit is None or dim != 0 or self.world_size == 1:
            return None
        chunk_size = limit // input_.element_size()
        return chunk_size if input_.numel() > chunk_size * shards else None

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        chunk_size = self._sequence_chunk_size(input_, dim)
        gather = super().all_gather
        if chunk_size is None:
            return gather(input_, dim)

        chunks = []
        for chunk in input_.reshape(-1).split(chunk_size):
            chunks.append(gather(chunk, dim=0).view(self.world_size, -1))
        return torch.cat(chunks, dim=1).reshape(self.world_size * input_.shape[0], *input_.shape[1:])

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        chunk_size = self._sequence_chunk_size(input_, dim, self.world_size)
        scatter = super().reduce_scatter
        if chunk_size is None:
            return scatter(input_, dim)

        assert input_.shape[0] % self.world_size == 0
        # Split the same offset range out of every destination's shard.
        shards = input_.reshape(self.world_size, -1)
        chunks = []
        for chunk in shards.split(chunk_size, dim=1):
            chunks.append(scatter(chunk.contiguous().view(-1), dim=0))
        return torch.cat(chunks).reshape(input_.shape[0] // self.world_size, *input_.shape[1:])
