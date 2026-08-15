# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
import torch.distributed as dist
from vllm.distributed.eplb.eplb_communicator import TorchDistNcclEplbCommunicator


class HcclEplbCommunicator(TorchDistNcclEplbCommunicator):
    """Torch-distributed EPLB transfers over the HCCL device group."""

    def _to_global_peer_rank(self, peer_group_rank: int) -> int:
        """Translate an EPLB group-local peer rank for upstream P2POp.

        EPLB transfer planning uses ranks local to the EPLB process group,
        while the current upstream torch-distributed communicator passes its
        peer argument to ``P2POp`` as a global rank. These rank spaces differ
        for pipeline stages whose EPLB groups do not start at global rank zero.

        TODO: Remove this compatibility adapter after the upstream
        communicator passes local peers through ``P2POp(group_peer=...)``.
        """
        group_size = self._ep_group.size()
        if not 0 <= peer_group_rank < group_size:
            raise ValueError(f"EPLB peer group rank {peer_group_rank} is outside the valid range [0, {group_size}).")
        return dist.get_global_rank(self._ep_group, peer_group_rank)

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
        # Ascend keeps each expert in an independent persistent tensor. The
        # upstream profile collective expects every weight entry to be one
        # stacked tensor, so reserve HCCL buffers during actual P2P transfers.
        return False
