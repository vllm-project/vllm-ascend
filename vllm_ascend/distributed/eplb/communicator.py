# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
import torch.distributed as dist
from vllm.distributed.eplb.eplb_communicator import TorchDistGlooStagedEplbCommunicator


class AscendGlooEplbCommunicator(TorchDistGlooStagedEplbCommunicator):
    """Gloo CPU-staging EPLB communicator for async mode on Ascend.

    Gloo uses CPU-side P2P and does not require the NCCL/HCCL buffer
    reservation collective that the upstream profile path runs. Disabling
    it also avoids passing Ascend's EplbExpertTensorList to all_gather,
    which does not implement the __torch_function__ protocol for
    distributed collectives.
    """

    def _to_global_peer_rank(self, peer_group_rank: int) -> int:
        """Translate an EPLB group-local peer rank for torch.distributed."""
        group_size = self._cpu_group.size()
        if not 0 <= peer_group_rank < group_size:
            raise ValueError(f"EPLB peer group rank {peer_group_rank} is outside the valid range [0, {group_size}).")
        return dist.get_global_rank(self._cpu_group, peer_group_rank)

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        super().add_send(tensors, self._to_global_peer_rank(dst_rank), expert_id)

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        super().add_recv(tensors, self._to_global_peer_rank(src_rank), expert_id)

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        return False
