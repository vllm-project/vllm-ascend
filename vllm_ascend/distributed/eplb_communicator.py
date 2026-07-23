# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext

import torch
from torch.distributed import P2POp, ProcessGroup, batch_isend_irecv
from vllm.distributed.eplb.eplb_communicator import EplbCommunicator


class HcclEplbCommunicator(EplbCommunicator):
    """EPLB expert-weight communicator backed by the HCCL device group."""

    def __init__(self, ep_group: ProcessGroup) -> None:
        self._ep_group = ep_group
        self._npu_stream = None
        self._p2p_ops: list[P2POp] = []
        self._recv_staging: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._log_initialized()

    @staticmethod
    def _requires_staging(tensor: torch.Tensor) -> bool:
        # HCCL rejects non-zero storage offsets for internal-format tensors.
        # Expert rows are contiguous views, so contiguous() would not allocate
        # independent storage; clone()/empty_like() are required here.
        return tensor.storage_offset() != 0

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        del expert_id
        for tensor in tensors:
            send_tensor = tensor.clone() if self._requires_staging(tensor) else tensor
            self._p2p_ops.append(
                P2POp(
                    torch.distributed.isend,
                    send_tensor,
                    dst_rank,
                    self._ep_group,
                )
            )

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        del expert_id
        for tensor in tensors:
            if self._requires_staging(tensor):
                recv_tensor = torch.empty_like(tensor)
                self._recv_staging.append((tensor, recv_tensor))
            else:
                recv_tensor = tensor
            self._p2p_ops.append(
                P2POp(
                    torch.distributed.irecv,
                    recv_tensor,
                    src_rank,
                    self._ep_group,
                )
            )

    def execute(self) -> None:
        if not self._p2p_ops:
            return
        stream_context = nullcontext() if self._npu_stream is None else torch.npu.stream(self._npu_stream)
        try:
            with stream_context:
                requests = batch_isend_irecv(self._p2p_ops)
                for request in requests:
                    request.wait()
                for target, staging in self._recv_staging:
                    target.copy_(staging)
        finally:
            self._p2p_ops.clear()
            self._recv_staging.clear()

    def set_stream(self, npu_stream) -> None:
        self._npu_stream = npu_stream
