# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm.distributed.eplb.eplb_communicator import TorchDistNcclEplbCommunicator


class HcclEplbCommunicator(TorchDistNcclEplbCommunicator):
    """Torch-distributed EPLB transfers over the HCCL device group."""

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        # Ascend keeps each expert in an independent persistent tensor. The
        # upstream profile collective expects every weight entry to be one
        # stacked tensor, so reserve HCCL buffers during actual P2P transfers.
        return False
