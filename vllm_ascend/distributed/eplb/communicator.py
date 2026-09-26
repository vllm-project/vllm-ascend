# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
import torch.distributed as dist
from torch.distributed import P2POp, batch_isend_irecv
from vllm.distributed.eplb.eplb_communicator import TorchDistGlooStagedEplbCommunicator
from vllm.distributed.eplb.eplb_utils import device_stream
from vllm.utils.gpu_sync_debug import gpu_sync_allowed


class AscendGlooEplbCommunicator(TorchDistGlooStagedEplbCommunicator):
    """Gloo CPU-staging EPLB communicator for async mode on Ascend.

    Gloo uses CPU-side P2P and does not require the NCCL/HCCL buffer
    reservation collective that the upstream profile path runs. Disabling
    it also avoids passing Ascend's EplbExpertTensorList to all_gather,
    which does not implement the __torch_function__ protocol for
    distributed collectives.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pinned_staging_buffers: dict[tuple[torch.dtype, tuple[int, ...]], list[torch.Tensor]] = {}

    def _acquire_staging_buffer(
        self,
        tensor: torch.Tensor,
        buffer_indices: dict[tuple[torch.dtype, tuple[int, ...]], int],
    ) -> torch.Tensor:
        key = tensor.dtype, tuple(tensor.shape)
        buffer_index = buffer_indices.get(key, 0)
        buffers = self._pinned_staging_buffers.setdefault(key, [])
        if buffer_index == len(buffers):
            buffers.append(torch.empty_like(tensor, device="cpu", pin_memory=True))
        buffer_indices[key] = buffer_index + 1
        return buffers[buffer_index]

    def execute(self) -> None:
        if not self._ops:
            return

        stream = self._stream
        p2p_ops: list[P2POp] = []
        recv_staging: list[tuple[torch.Tensor, torch.Tensor]] = []
        buffer_indices: dict[tuple[torch.dtype, tuple[int, ...]], int] = {}
        try:
            with device_stream(stream):
                for operation, tensor, peer_rank in self._ops:
                    cpu_tensor = self._acquire_staging_buffer(tensor, buffer_indices)
                    if operation == "send":
                        cpu_tensor.copy_(tensor, non_blocking=True)
                        p2p_ops.append(
                            P2POp(
                                dist.isend,
                                cpu_tensor,
                                group=self._cpu_group,
                                group_peer=peer_rank,
                            )
                        )
                    else:
                        p2p_ops.append(
                            P2POp(
                                dist.irecv,
                                cpu_tensor,
                                group=self._cpu_group,
                                group_peer=peer_rank,
                            )
                        )
                        recv_staging.append((tensor, cpu_tensor))
        finally:
            self._ops.clear()

        with gpu_sync_allowed():
            if stream is not None:
                stream.synchronize()
            else:
                torch.accelerator.current_stream().synchronize()

        for request in batch_isend_irecv(p2p_ops):
            request.wait()

        with device_stream(stream):
            for dst_tensor, cpu_tensor in recv_staging:
                dst_tensor.copy_(cpu_tensor, non_blocking=True)

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        return False
