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

    def _hccl_backend(self):
        """Return the ProcessGroupHCCL backend of the device group.

        vLLM's stateless groups are registered in torch.distributed by
        name only (not in ``_world.pg_group_ranks``), so
        ``torch.distributed.send/recv/broadcast/isend/irecv`` would raise
        "Group ... is not registered" -- and would also misinterpret
        group-local ranks as global ranks. Calling the backend directly
        bypasses both, mirroring how CudaCommunicator routes P2P and
        broadcast through pynccl.
        """
        if self.device_group is None:
            raise ValueError("No HCCL device group found")
        return self.device_group._get_backend(torch.device("npu"))

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        # pg bind signature: send(tensors: Sequence[Tensor], dst, tag)
        self._hccl_backend().send([tensor], dst, 0).wait()

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        self._hccl_backend().recv([tensor], src, 0).wait()
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all ranks."""
        if self.world_size == 1:
            return tensor
        # pg bind overload: broadcast(tensor: Tensor, root) -> Work
        self._hccl_backend().broadcast(tensor, src).wait()
        return tensor

    def batch_isend_irecv(self, p2p_ops: list):
        """Batch point-to-point ops over the (possibly stateless) HCCL group.

        Delegates to ProcessGroupHCCL's native
        ``batch_isend_irecv(op_types, tensors, remote_ranks)``. Manually
        looping async ``send``/``recv`` calls does NOT work: each op lazily
        creates its own HCCL comm via rootInfo exchange, and with more than
        one op in flight (any bidirectional pattern, or a ring exchange)
        that per-op comm creation fails with EI0015 (rootInfo detect
        timeout) or EI0005 (inconsistent arguments) -- verified on
        910 with the netloader-style stateless groups, where single
        unidirectional send/recv pairs work but manual batches never do.
        The native batch API creates one comm for the whole batch, which
        is the HCCL equivalent of ncclGroupStart/End batching used by
        CudaCommunicator.batch_isend_irecv.

        Ops carry group-local peer ranks (``op.group_peer``), matching the
        native API's rank semantics; stateless groups are not in
        torch.distributed's ``_world`` registry, so
        torch.distributed.batch_isend_irecv would raise "Group ... is not
        registered" anyway.
        """
        backend = self._hccl_backend()
        op_types: list[str] = []
        tensors: list[torch.Tensor] = []
        remote_ranks: list[int] = []
        for op in p2p_ops:
            if op.op is torch.distributed.isend:
                op_types.append("isend")
                # HCCL reads the send buffer as-is; hand it contiguous
                # memory (no-op for already-contiguous tensors).
                tensors.append(op.tensor.contiguous())
            elif op.op is torch.distributed.irecv:
                op_types.append("irecv")
                # Never copy recv buffers: data must land in op.tensor.
                tensors.append(op.tensor)
            else:
                raise ValueError(f"Unsupported p2p op: {op.op}")
            remote_ranks.append(op.group_peer)
        backend.batch_isend_irecv(op_types, tensors, remote_ranks).wait()
