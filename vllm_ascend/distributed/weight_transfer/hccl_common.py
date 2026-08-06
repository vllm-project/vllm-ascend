# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared HCCL initialization helpers for weight transfer engines."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.distributed.weight_transfer.base import WeightTransferInitInfo

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig

    from vllm_ascend.distributed.device_communicators.pyhccl import (
        PyHcclCommunicator,
    )


@dataclass
class HCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for HCCL-based weight transfer backends."""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device,
) -> "PyHcclCommunicator":
    """Create a stateless HCCL process group for weight transfer."""
    from vllm.distributed.utils import StatelessProcessGroup

    from vllm_ascend.distributed.device_communicators.pyhccl import (
        PyHcclCommunicator,
    )

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size,
    )
    return PyHcclCommunicator(pg, device=device)


def worker_init_process_group(
    init_info: HCCLWeightTransferInitInfo,
    parallel_config: "ParallelConfig",
) -> "PyHcclCommunicator":
    """Create the trainer-to-worker HCCL group on an inference worker."""
    dp_rank = parallel_config.data_parallel_index
    world_size_per_dp = parallel_config.world_size
    rank_within_dp = parallel_config.rank
    worker_rank = dp_rank * world_size_per_dp + rank_within_dp
    rank = worker_rank + init_info.rank_offset
    device = torch.accelerator.current_device_index()
    return stateless_init_process_group(
        init_info.master_address,
        init_info.master_port,
        rank,
        init_info.world_size,
        device=device,
    )


def trainer_init(
    init_info: HCCLWeightTransferInitInfo | dict,
) -> "PyHcclCommunicator":
    """Create the HCCL group for trainer rank zero."""
    if isinstance(init_info, dict):
        master_address = init_info["master_address"]
        master_port = init_info["master_port"]
        world_size = init_info["world_size"]
    else:
        master_address = init_info.master_address
        master_port = init_info.master_port
        world_size = init_info.world_size

    device = torch.accelerator.current_device_index()
    return stateless_init_process_group(
        master_address,
        master_port,
        0,
        world_size,
        device,
    )
