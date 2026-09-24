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
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        use_all2all: bool = False,
        tcp_store_group=None,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            use_all2all=use_all2all,
        )
        # TODO(hz): Refer to CudaCommunicator's implementation to integrate PyHcclCommunicator
        # init device according to rank
        # NOTE: keep the legacy behavior (bare device index) when no device
        # is passed by the regular (non-stateless) GroupCoordinator path.
        # StatelessGroupCoordinator passes a proper torch.device("npu:N").
        if device is None:
            self.device = torch.npu.current_device()
        # `tcp_store_group` is accepted for signature compatibility with
        # CudaCommunicator (StatelessGroupCoordinator passes it when
        # building standby DP/EP groups for elastic EP); HCCL P2P goes
        # through the device_group directly, so it is not stored.

        self.ca_comm = None
        # vLLM #53576 reads this CUDA-only communicator during graph capture.
        # Keep the shared coordinator protocol available without enabling the
        # FlashInfer PCIe IPC backend on NPU.
        self.fi_pcie_ipc_ar_comm = None
        self.all2all_manager = _NpuAll2AllManager()

    def batch_isend_irecv(self, p2p_ops: list):
        """Batch point-to-point ops over the (possibly stateless) HCCL group.

        Mirrors CudaCommunicator.batch_isend_irecv, which batches ops
        between ncclGroupStart/End. ProcessGroupHCCL has no group-start/end
        batching, so we post all ops as async work first and wait at the
        end, which preserves the non-deadlocking semantics of a batch
        (both directions can be in flight simultaneously).
        """
        handles = []
        for op in p2p_ops:
            if op.op is torch.distributed.isend:
                handles.append(
                    torch.distributed.isend(
                        op.tensor, op.group_peer, self.device_group
                    )
                )
            elif op.op is torch.distributed.irecv:
                handles.append(
                    torch.distributed.irecv(
                        op.tensor, op.group_peer, self.device_group
                    )
                )
            else:
                raise ValueError(f"Unsupported p2p op: {op.op}")
        for handle in handles:
            handle.wait()
